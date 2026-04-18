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
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from encode import encode, initial_state, legal_mask
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
) -> list[Trajectory]:
    model.eval()

    games = [Game(size=5) for _ in range(num_games)]
    states = [initial_state(g) for g in games]
    trajectories: list[Trajectory] = [Trajectory() for _ in range(num_games)]
    active = [True] * num_games

    for _ in range(max_steps):
        active_indices = [i for i, a in enumerate(active) if a]
        if not active_indices:
            break

        batch_tensors = [
            encode(states[i], states[i].turn, states[i].phase is Phase.PLACEMENT)
            for i in active_indices
        ]
        batch_masks = [legal_mask(states[i]) for i in active_indices]

        batch = torch.stack(batch_tensors).to(device, non_blocking=True)
        masks = torch.stack(batch_masks).to(device, non_blocking=True)

        if device.type == "cuda":
            batch = batch.to(memory_format=torch.channels_last)

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
                    state_tensor=batch_tensors[slot],
                    legal=batch_masks[slot],
                    action=a,
                    player=s.turn,
                )
            )
            new_state = games[i].click_tile(a % 5, a // 5)
            states[i] = new_state
            if new_state.phase is Phase.ENDED or not new_state.accepted:
                trajectories[i].winner = new_state.winner
                active[i] = False

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

    if device.type == "cuda":
        states = states.to(memory_format=torch.channels_last)

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

        advantage = outcomes - values.detach()
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-iter", type=int, default=512)
    parser.add_argument("--iters", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs-per-iter", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--ckpt-every", type=int, default=25)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = pick_device()
    tune_runtime(device)
    use_amp = (device.type == "cuda") and (not args.no_amp)
    print(f"device={device} amp={use_amp} compile={args.compile}")

    model = ChainReactionNet(
        in_channels=4, channels=args.channels, num_blocks=args.blocks
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model, mode="max-autotune")

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
        )
        t_play = time.time() - t0

        finished = [t for t in trajs if t.winner is not None]
        n_p1 = sum(1 for t in finished if t.winner is True)
        avg_len = sum(len(t.steps) for t in trajs) / max(len(trajs), 1)

        batch = trajectories_to_batch(trajs, device)
        if batch is None:
            print(f"iter {it:05d} | no finished games -- skipping update")
            continue

        states, masks, actions, outcomes = batch
        n_samples = states.size(0)

        t1 = time.time()
        losses: list[dict[str, float]] = []
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
        t_train = time.time() - t1

        avg = {k: sum(l[k] for l in losses) / len(losses) for k in losses[0]}
        print(
            f"iter {it:05d} | games {len(trajs)} (done {len(finished)}, "
            f"p1_wins {n_p1}) | samples {n_samples} avg_len {avg_len:.1f} | "
            f"loss {avg['loss']:.3f} pl {avg['policy_loss']:.3f} "
            f"vl {avg['value_loss']:.3f} H {avg['entropy']:.3f} | "
            f"play {t_play:.1f}s train {t_train:.1f}s"
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


if __name__ == "__main__":
    main()
