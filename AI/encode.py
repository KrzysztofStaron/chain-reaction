from __future__ import annotations

import torch

from game import Game, GameState, Phase, Tile


def calc_dif(board: GameState, current_player: bool) -> float:
    player_count = 0
    opp_count = 0
    for tile in board.tiles:
        if tile.player is None:
            continue
        if tile.player == current_player:
            player_count += 1
        else:
            opp_count += 1
    return (player_count - opp_count) / 25.0  # in [-1.0, 1.0]


def encode(board: GameState, current_player: bool, is_placement_phase: bool) -> torch.Tensor:
    """Encode a board from `current_player`'s point of view into [4, 5, 5]."""
    tensor = torch.zeros((4, 5, 5), dtype=torch.float32)

    for i in range(25):
        y = i // 5
        x = i % 5
        tile = board.tile_at(x, y)

        if tile.player is not None:
            tensor[0, y, x] = tile.value / 3.0  # placement orbs sit at 3, stable cells <= 3
            tensor[1, y, x] = 1.0 if tile.player == current_player else -1.0

    tensor[2] = 1.0 if is_placement_phase else 0.0
    tensor[3] = calc_dif(board, current_player)
    return tensor


def legal_mask(state: GameState) -> torch.Tensor:
    """Return a [25] bool mask of legal moves for `state.turn`.

    Placement phase: any unowned cell is legal.
    Playing phase:   only cells owned by the current player are legal.
    """
    mask = torch.zeros(25, dtype=torch.bool)
    if state.phase is Phase.PLACEMENT:
        for i, t in enumerate(state.tiles):
            mask[i] = t.player is None
    else:
        for i, t in enumerate(state.tiles):
            mask[i] = t.player == state.turn
    return mask


def initial_state(game: Game) -> GameState:
    """Return the freshly-constructed snapshot of `game` without mutating it."""
    # Out-of-bounds click is rejected but still returns the current snapshot.
    return game.click_tile(-1, -1)


__all__ = ["calc_dif", "encode", "legal_mask", "initial_state", "Tile"]
