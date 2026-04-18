from __future__ import annotations

import torch

from game import Game, GameState, Phase, Tile


def calc_dif(board: GameState, current_player: bool) -> float:
    player_count = 0
    opp_count = 0
    for p in board.players:
        if p is None:
            continue
        if p == current_player:
            player_count += 1
        else:
            opp_count += 1
    return (player_count - opp_count) / 25.0  # in [-1.0, 1.0]


def encode(board: GameState, current_player: bool, is_placement_phase: bool) -> torch.Tensor:
    """Encode a board from `current_player`'s point of view into [4, 5, 5]."""
    vals = board.values
    plrs = board.players
    phase_val = 1.0 if is_placement_phase else 0.0
    dif = calc_dif(board, current_player)

    # Build flat list then create tensor in one C call — avoids 25+ slow
    # Python→torch element assignments.
    ch0 = [0.0] * 25
    ch1 = [0.0] * 25
    for i in range(25):
        p = plrs[i]
        if p is not None:
            ch0[i] = vals[i] / 3.0
            ch1[i] = 1.0 if p == current_player else -1.0

    data = ch0 + ch1 + [phase_val] * 25 + [dif] * 25
    return torch.tensor(data, dtype=torch.float32).view(4, 5, 5)


def legal_mask(state: GameState) -> torch.Tensor:
    """Return a [25] bool mask of legal moves for `state.turn`.

    Placement phase: any unowned cell is legal.
    Playing phase:   only cells owned by the current player are legal.
    """
    plrs = state.players
    if state.phase is Phase.PLACEMENT:
        return torch.tensor([p is None for p in plrs], dtype=torch.bool)
    turn = state.turn
    return torch.tensor([p == turn for p in plrs], dtype=torch.bool)


def batch_encode_and_mask(states: list[GameState]) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode multiple game states into a single batch.

    Returns (batch, masks) where:
        batch: [N, 4, 5, 5] float32
        masks: [N, 25] bool
    Creates exactly 2 tensors total instead of 2*N, avoiding per-state
    tensor allocation and the torch.stack() copy.
    """
    all_enc: list[float] = []
    all_mask: list[bool] = []

    for s in states:
        vals = s.values
        plrs = s.players
        cp = s.turn
        is_place = s.phase is Phase.PLACEMENT
        phase_val = 1.0 if is_place else 0.0

        player_count = 0
        opp_count = 0

        ch0 = [0.0] * 25
        ch1 = [0.0] * 25
        for i in range(25):
            p = plrs[i]
            if p is not None:
                ch0[i] = vals[i] / 3.0
                ch1[i] = 1.0 if p == cp else -1.0
                if p == cp:
                    player_count += 1
                else:
                    opp_count += 1

        dif = (player_count - opp_count) / 25.0
        all_enc.extend(ch0)
        all_enc.extend(ch1)
        for _ in range(25):
            all_enc.append(phase_val)
        for _ in range(25):
            all_enc.append(dif)

        if is_place:
            all_mask.extend(p is None for p in plrs)
        else:
            all_mask.extend(p == cp for p in plrs)

    n = len(states)
    batch = torch.tensor(all_enc, dtype=torch.float32).view(n, 4, 5, 5)
    masks = torch.tensor(all_mask, dtype=torch.bool).view(n, 25)
    return batch, masks


def initial_state(game: Game) -> GameState:
    """Return the freshly-constructed snapshot of `game` without mutating it."""
    return game.click_tile(-1, -1)


__all__ = [
    "calc_dif", "encode", "legal_mask", "batch_encode_and_mask",
    "initial_state", "Tile",
]
