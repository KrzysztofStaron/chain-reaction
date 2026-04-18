"""
Self-play + training for Chain Reaction.

Design:
    - N games played in parallel in Python; each step we batch the active games'
      encoded states through the network on the GPU, mask illegal moves, sample
      actions, and step the game engine forward. Tiny 5x5 game logic stays on
      CPU -- it's extremely cheap -- while the NN crunches big batches on GPU.
    - Training signal is REINFORCE with a value baseline (AlphaZero-lite,
      without MCTS): z in {-1, +1} from the point of view of the player who
      acted, policy gets -log pi(a|s) * (z - V.detach()), value gets MSE(V, z),
      plus a small entropy bonus to keep exploring.
    - Checkpoints every --ckpt-every iterations (plus a rolling "latest.pt"),
      resumable with --resume.

High-end GPU knobs (all on by default when CUDA is present):
    - bfloat16 autocast in forward + loss
    - TF32 matmul / cuDNN benchmark
    - channels-last memory format for convs
    - optional torch.compile via --compile
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from huggingface_hub import HfApi
from torch.optim import AdamW
from tqdm import tqdm

from encode import batch_encode_and_mask, encode, initial_state, legal_mask
from game import Game, Phase
from play import ChainReactionNet


# ----------------------------------------------------------------------------
# Device + runtime tuning
# ----------------------------------------------------------------------------
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tune_runtime(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


# ----------------------------------------------------------------------------
# Trajectory containers
# ----------------------------------------------------------------------------
@dataclass
class Step:
    state_tensor: torch.Tensor  # [C, 5, 5], CPU float32
    legal: torch.Tensor         # [25] bool, CPU
    action: int
    player: bool                # who acted at this step


@dataclass
class Trajectory:
    steps: list[Step] = field(default_factory=list)
    winner: bool | None = None  # None means the game never finished


# ----------------------------------------------------------------------------
# Batched self-play
# ----------------------------------------------------------------------------
@torch.no_grad()
def run_selfplay_batch(
    model: ChainReactionNet,
    num_games: int,
    device: torch.device,
    max_steps: int = 200,
    temperature: float = 1.0,
    use_amp: bool = False,
    show_progress: bool = True,
) -> list[Trajectory]:
    model.eval()

    games = [Game(size=5) for _ in range(num_games)]
    states = [initial_state(g) for g in games]
    trajectories: list[Trajectory] = [Trajectory() for _ in range(num_games)]
    active = [True] * num_games

    pbar = tqdm(
        total=num_games,
        desc="self-play",
        unit="game",
        leave=False,
        disable=not show_progress,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
    prev_done = 0
    try:
        for step in range(max_steps):
            active_indices = [i for i, a in enumerate(active) if a]
            if not active_indices:
                break

            active_states = [states[i] for i in active_indices]
            batch_cpu, masks_cpu = batch_encode_and_mask(active_states)

            if device.type == "cuda":
                batch_cpu = batch_cpu.pin_memory()
                masks_cpu = masks_cpu.pin_memory()

            batch = batch_cpu.to(device, non_blocking=True)
            masks = masks_cpu.to(device, non_blocking=True)

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp and device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with amp_ctx:
                logits, _ = model(batch)

            logits = logits.float().masked_fill(~masks, float("-inf"))

            if temperature <= 0.0:
                actions_t = logits.argmax(dim=-1)
            else:
                # Gumbel-max trick: argmax(logits/T + Gumbel) ~ Categorical(softmax(logits/T)).
                # Avoids torch.multinomial (buggy on MPS) and softmax-of-(-inf) rows
                # returning NaN on some MPS builds. -inf + finite stays -inf, so
                # illegal moves can never win the argmax.
                u = torch.rand_like(logits).clamp_(min=1e-20, max=1.0 - 1e-7)
                gumbel = -torch.log(-torch.log(u))
                actions_t = (logits / temperature + gumbel).argmax(dim=-1)

            actions = actions_t.cpu().tolist()

            for slot, i in enumerate(active_indices):
                a = actions[slot]
                s = states[i]
                trajectories[i].steps.append(
                    Step(
                        # Clone rows so we do not retain every timestep's full
                        # [N,4,5,5] batch tensor in memory (was blowing up RAM/GC).
                        state_tensor=batch_cpu[slot].clone(),
                        legal=masks_cpu[slot].clone(),
                        action=a,
                        player=s.turn,
                    )
                )
                new_state = games[i].click_tile(a % 5, a // 5)
                states[i] = new_state
                if new_state.phase is Phase.ENDED or not new_state.accepted:
                    trajectories[i].winner = new_state.winner
                    active[i] = False

            n_done = num_games - sum(active)
            if n_done > prev_done:
                pbar.update(n_done - prev_done)
                prev_done = n_done
            pbar.set_postfix_str(f"active={sum(active)} step={step}")

    finally:
        pbar.close()

    return trajectories


# ----------------------------------------------------------------------------
# Trajectory -> training tensors
# ----------------------------------------------------------------------------
def trajectories_to_batch(trajectories: list[Trajectory], device: torch.device):
    states: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    actions: list[int] = []
    outcomes: list[float] = []

    for traj in trajectories:
        if traj.winner is None:
            continue
        for step in traj.steps:
            z = 1.0 if traj.winner == step.player else -1.0
            states.append(step.state_tensor)
            masks.append(step.legal)
            actions.append(step.action)
            outcomes.append(z)

    if not states:
        return None

    states_t = torch.stack(states).to(device, non_blocking=True)
    masks_t = torch.stack(masks).to(device, non_blocking=True)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    outcomes_t = torch.tensor(outcomes, dtype=torch.float32, device=device)
    return states_t, masks_t, actions_t, outcomes_t


# ----------------------------------------------------------------------------
# D4 symmetry augmentation
# ----------------------------------------------------------------------------
# The 5x5 board is invariant under the dihedral group D4: 4 rotations times
# {identity, mirror} = 8 symmetries. For each sample (state, mask, action, z)
# we produce 8 equivalent samples by rotating/flipping the board and remapping
# the action index through the same permutation. z is orientation-invariant.
_SYM_PERMS_CACHE: dict[int, torch.Tensor] = {}


def _sym_perms(size: int, device: torch.device) -> torch.Tensor:
    """Return an [8, size*size] long tensor: perm[k, i] is the new flat index
    for cell i under the k-th symmetry. Cached per (size, device)."""
    cache_key = (size, device.type, device.index if device.index is not None else -1)
    cached = _SYM_PERMS_CACHE.get(hash(cache_key))
    if cached is not None:
        return cached

    transforms = [
        lambda y, x: (y, x),                          # 0: identity
        lambda y, x: (size - 1 - x, y),               # 1: rot90 CCW
        lambda y, x: (size - 1 - y, size - 1 - x),    # 2: rot180
        lambda y, x: (x, size - 1 - y),               # 3: rot270 CCW
        lambda y, x: (y, size - 1 - x),               # 4: flip x (mirror L-R)
        lambda y, x: (size - 1 - y, x),               # 5: flip y (mirror U-D)
        lambda y, x: (x, y),                          # 6: transpose (main diag)
        lambda y, x: (size - 1 - x, size - 1 - y),    # 7: anti-transpose
    ]
    perms = torch.zeros((8, size * size), dtype=torch.long)
    for k, t in enumerate(transforms):
        for y in range(size):
            for x in range(size):
                ny, nx = t(y, x)
                perms[k, y * size + x] = ny * size + nx
    perms = perms.to(device)
    _SYM_PERMS_CACHE[hash(cache_key)] = perms
    return perms


def _rotate_board(tensor: torch.Tensor, sym_idx: int) -> torch.Tensor:
    """Apply the k-th D4 symmetry to a board-shaped tensor (..., 5, 5)."""
    if sym_idx == 0:
        return tensor
    if sym_idx == 1:
        return tensor.rot90(k=1, dims=(-2, -1))
    if sym_idx == 2:
        return tensor.rot90(k=2, dims=(-2, -1))
    if sym_idx == 3:
        return tensor.rot90(k=3, dims=(-2, -1))
    if sym_idx == 4:
        return tensor.flip(dims=(-1,))
    if sym_idx == 5:
        return tensor.flip(dims=(-2,))
    if sym_idx == 6:
        return tensor.transpose(-2, -1)
    # sym_idx == 7: anti-transpose == rot180 then transpose
    return tensor.rot90(k=2, dims=(-2, -1)).transpose(-2, -1)


def augment_symmetries(
    states: torch.Tensor,    # [N, C, 5, 5]
    masks: torch.Tensor,     # [N, 25] bool
    actions: torch.Tensor,   # [N] long
    outcomes: torch.Tensor,  # [N] float
    size: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand a batch by the 8 D4 symmetries of the board (identity included).

    Tensor rotations and the action-index permutation are kept consistent:
    rotating state[..., y, x] by symmetry k sends the value to (y', x') via
    transform k; action index y*N+x is remapped to y'*N+x' via the same k.
    """
    perms = _sym_perms(size, states.device)  # [8, size*size]

    out_states: list[torch.Tensor] = []
    out_masks: list[torch.Tensor] = []
    out_actions: list[torch.Tensor] = []
    out_outcomes: list[torch.Tensor] = []

    mask_2d = masks.view(-1, size, size)

    for k in range(8):
        rot_states = _rotate_board(states, k).contiguous()
        rot_mask_2d = _rotate_board(mask_2d, k).contiguous()
        rot_masks = rot_mask_2d.view(-1, size * size)
        rot_actions = perms[k][actions]

        out_states.append(rot_states)
        out_masks.append(rot_masks)
        out_actions.append(rot_actions)
        out_outcomes.append(outcomes)

    return (
        torch.cat(out_states, dim=0),
        torch.cat(out_masks, dim=0),
        torch.cat(out_actions, dim=0),
        torch.cat(out_outcomes, dim=0),
    )


# ----------------------------------------------------------------------------
# Training step
# ----------------------------------------------------------------------------
def train_step(
    model: ChainReactionNet,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    use_amp: bool,
    device: torch.device,
    entropy_coef: float,
    value_coef: float,
) -> dict[str, float]:
    states, masks, actions, outcomes = batch
    model.train()

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp and device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with amp_ctx:
        logits, values = model(states)
        logits = logits.float()
        # MPS log_softmax returns NaN when a row contains -inf, even though the
        # math is well-defined. Use a very-negative finite sentinel instead.
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(~masks, neg_inf)
        logp = F.log_softmax(masked_logits, dim=-1)
        logp_a = logp.gather(1, actions.unsqueeze(1)).squeeze(1)

        raw_adv = outcomes - values.detach()
        advantage = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-8)
        policy_loss = -(logp_a * advantage).mean()
        value_loss = F.mse_loss(values, outcomes)

        # Entropy only over legal moves (illegal probs underflow to 0 but we
        # also zero them explicitly to avoid 0 * huge_negative = NaN).
        legal_f = masks.float()
        probs = logp.exp() * legal_f
        entropy = -(probs * logp * legal_f).sum(dim=-1).mean()

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "entropy": float(entropy.detach().item()),
    }


# ----------------------------------------------------------------------------
# Main training loop
# ----------------------------------------------------------------------------
def main() -> None:
    _cuda = torch.cuda.is_available()
    # CUDA: larger defaults to feed the GPU (tiny 5×5 convs are launch-bound at 2k batch).
    _def_games = 1536 if _cuda else 512
    _def_batch = 12288 if _cuda else 2048

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--games-per-iter",
        type=int,
        default=_def_games,
        help=f"Parallel self-play envs (default {_def_games} on CUDA, 512 on CPU/MPS).",
    )
    parser.add_argument("--iters", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs-per-iter", type=int, default=4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_def_batch,
        help=f"Train minibatch size (default {_def_batch} on CUDA, 2048 elsewhere). "
        "On 10–12GB try 16384–24576 if OOM is not hit.",
    )
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.03)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--ckpt-every", type=int, default=25)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the policy net (CUDA: often +10–40%% train throughput; "
        "first iter pays compile cost).",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=("default", "reduce-overhead", "max-autotune"),
        help="torch.compile mode (reduce-overhead fits small models well).",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable 8x D4 symmetry augmentation of the training batch.",
    )
    parser.add_argument("--wandb-project", type=str, default="chain-reaction")
    parser.add_argument("--hf-repo", type=str, default="PanzerBread/chain-reaction")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-hf", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm self-play progress bar (cleaner logs / non-TTY).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = pick_device()
    tune_runtime(device)
    use_amp = (device.type == "cuda") and (not args.no_amp)
    print(
        f"device={device} amp={use_amp} compile={args.compile} "
        f"(games={args.games_per_iter} batch={args.batch_size})"
    )
    if device.type == "cuda":
        print(
            "CUDA saturation: this model is tiny — if util/memory stay low, raise "
            "--batch-size (e.g. 24576) and/or --games-per-iter; add --compile for "
            "faster train steps. Self-play stays CPU-bound between NN forwards."
        )
    if device.type == "cpu" and args.games_per_iter >= 128:
        print(
            "warning: self-play is mostly batched NN forwards; on CPU this is often "
            "100–300s+ per iter at --games-per-iter 512. Use CUDA (or MPS on Mac) "
            "for ~10–30x faster play."
        )

    # ---- Weights & Biases ---------------------------------------------------
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            save_code=True,
        )
        print(f"wandb run: {wandb.run.url}")

    # ---- Hugging Face Hub ----------------------------------------------------
    use_hf = not args.no_hf
    hf_api = HfApi() if use_hf else None
    if use_hf:
        hf_api.create_repo(repo_id=args.hf_repo, exist_ok=True)
        print(f"HF repo: https://huggingface.co/{args.hf_repo}")

    model = ChainReactionNet(
        in_channels=4, channels=args.channels, num_blocks=args.blocks
    ).to(device)
    # Skip channels_last: for 5x5 boards the layout conversions cost more than
    # they save; also avoids a per-step batch permute in the self-play loop.
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_iter = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = int(ckpt["iter"]) + 1
        print(f"resumed from iter {start_iter - 1} ({args.resume})")

    for it in range(start_iter, args.iters):
        t0 = time.time()
        trajs = run_selfplay_batch(
            model,
            args.games_per_iter,
            device,
            max_steps=args.max_steps,
            temperature=args.temperature,
            use_amp=use_amp,
            show_progress=not args.no_progress,
        )
        t_play = time.time() - t0

        finished = [t for t in trajs if t.winner is not None]
        n_p1 = sum(1 for t in finished if t.winner is True)
        avg_len = sum(len(t.steps) for t in trajs) / max(len(trajs), 1)

        n_steps_collate = sum(len(t.steps) for t in finished)
        print(
            f"  iter {it:05d} | self-play {t_play:.1f}s — collating "
            f"{n_steps_collate} samples ({len(finished)}/{len(trajs)} games) → GPU…",
            flush=True,
        )
        batch = trajectories_to_batch(trajs, device)
        if batch is None:
            print(f"iter {it:05d} | no finished games -- skipping update")
            continue

        states, masks, actions, outcomes = batch
        raw_samples = states.size(0)
        if not args.no_augment:
            print(
                f"  iter {it:05d} | D4 augment 8× ({raw_samples} → {raw_samples * 8})…",
                flush=True,
            )
            states, masks, actions, outcomes = augment_symmetries(
                states, masks, actions, outcomes
            )
        n_samples = states.size(0)

        n_batch = (n_samples + args.batch_size - 1) // args.batch_size
        train_steps = args.epochs_per_iter * n_batch
        print(
            f"  iter {it:05d} | training {train_steps} minibatches "
            f"({args.epochs_per_iter}×{n_batch} @ {args.batch_size})…",
            flush=True,
        )

        t1 = time.time()
        losses: list[dict[str, float]] = []
        train_pbar = tqdm(
            total=train_steps,
            desc="train",
            unit="step",
            leave=False,
            disable=args.no_progress,
        )
        for _ in range(args.epochs_per_iter):
            perm = torch.randperm(n_samples, device=device)
            for s_idx in range(0, n_samples, args.batch_size):
                idx = perm[s_idx : s_idx + args.batch_size]
                sub = (states[idx], masks[idx], actions[idx], outcomes[idx])
                losses.append(
                    train_step(
                        model,
                        optimizer,
                        sub,
                        use_amp=use_amp,
                        device=device,
                        entropy_coef=args.entropy_coef,
                        value_coef=args.value_coef,
                    )
                )
                train_pbar.update(1)
        train_pbar.close()
        t_train = time.time() - t1

        avg = {k: sum(l[k] for l in losses) / len(losses) for k in losses[0]}
        samples_str = (
            f"samples {raw_samples}x8={n_samples}"
            if not args.no_augment
            else f"samples {n_samples}"
        )
        print(
            f"iter {it:05d} | games {len(trajs)} (done {len(finished)}, "
            f"p1_wins {n_p1}) | {samples_str} avg_len {avg_len:.1f} | "
            f"loss {avg['loss']:.3f} pl {avg['policy_loss']:.3f} "
            f"vl {avg['value_loss']:.3f} H {avg['entropy']:.3f} | "
            f"play {t_play:.1f}s train {t_train:.1f}s"
        )

        if use_wandb:
            wandb.log(
                {
                    "iter": it,
                    "loss": avg["loss"],
                    "policy_loss": avg["policy_loss"],
                    "value_loss": avg["value_loss"],
                    "entropy": avg["entropy"],
                    "games_finished": len(finished),
                    "games_total": len(trajs),
                    "p1_win_rate": n_p1 / max(len(finished), 1),
                    "avg_game_length": avg_len,
                    "samples_raw": raw_samples,
                    "samples_total": n_samples,
                    "time_selfplay": t_play,
                    "time_train": t_train,
                    "time_total": t_play + t_train,
                },
                step=it,
            )

        if (it + 1) % args.ckpt_every == 0:
            payload = {
                "iter": it,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            ckpt_path = ckpt_dir / f"ckpt_{it + 1:06d}.pt"
            torch.save(payload, ckpt_path)
            torch.save(payload, ckpt_dir / "latest.pt")
            print(f"  saved {ckpt_path}")

            if use_hf:
                hf_api.upload_file(
                    path_or_fileobj=str(ckpt_path),
                    path_in_repo=f"checkpoints/{ckpt_path.name}",
                    repo_id=args.hf_repo,
                    commit_message=f"checkpoint iter {it + 1}",
                )
                hf_api.upload_file(
                    path_or_fileobj=str(ckpt_dir / "latest.pt"),
                    path_in_repo="checkpoints/latest.pt",
                    repo_id=args.hf_repo,
                    commit_message=f"latest checkpoint (iter {it + 1})",
                )
                print(f"  uploaded to HF: {args.hf_repo}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
