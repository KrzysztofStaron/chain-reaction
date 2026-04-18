"""
Chain Reaction - pure-compute Python port.

The public surface of `Game` is intentionally tiny:
    - __init__(size=5)
    - click_tile(x, y) -> GameState

Everything else (chain-reaction propagation, win detection, phase
transitions) is computed synchronously inside `click_tile`. No
animation, no callbacks, no timers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional


BOARD_SIZE = 5
INITIAL_ORB_VALUE = 3


class Phase(str, Enum):
    PLACEMENT = "placement"
    PLAYING = "playing"
    ENDED = "ended"


# Player 1 = True, Player 2 = False, empty = None
Player = Optional[bool]


_OFFSETS: tuple[tuple[int, int], ...] = ((0, -1), (0, 1), (-1, 0), (1, 0))


@dataclass(frozen=True)
class Tile:
    value: int = 0
    player: Player = None


@dataclass(frozen=True)
class GameState:
    """Immutable snapshot returned by `click_tile`."""
    tiles: tuple[Tile, ...]
    size: int
    turn: bool
    phase: Phase
    winner: Player = None
    # True if the most recent click was accepted; False if ignored
    # (wrong phase, wrong player, ended, out of bounds, ...).
    accepted: bool = True

    def tile_at(self, x: int, y: int) -> Tile:
        return self.tiles[y * self.size + x]


class Game:
    def __init__(self, size: int = BOARD_SIZE, first_player: Optional[bool] = None) -> None:
        self._size = size
        self._tiles: list[Tile] = [Tile() for _ in range(size * size)]
        self._turn: bool = random.random() < 0.5 if first_player is None else first_player
        self._phase: Phase = Phase.PLACEMENT
        self._winner: Player = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def click_tile(self, x: int, y: int) -> GameState:
        if not self._in_bounds(x, y) or self._phase is Phase.ENDED:
            return self._snapshot(accepted=False)

        index = self._to_index(x, y)
        tile = self._tiles[index]

        if not self._can_interact(tile):
            return self._snapshot(accepted=False)

        if self._phase is Phase.PLACEMENT:
            self._place_initial(index)
            return self._snapshot(accepted=True)

        self._apply_click(index)
        self._run_chain_reaction()
        self._finalize_turn()
        return self._snapshot(accepted=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self._size and 0 <= y < self._size

    def _to_index(self, x: int, y: int) -> int:
        return y * self._size + x

    def _to_xy(self, index: int) -> tuple[int, int]:
        return index % self._size, index // self._size

    def _neighbors(self, index: int) -> list[int]:
        x, y = self._to_xy(index)
        result: list[int] = []
        for dx, dy in _OFFSETS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self._size and 0 <= ny < self._size:
                result.append(self._to_index(nx, ny))
        return result

    def _can_interact(self, tile: Tile) -> bool:
        if self._phase is Phase.PLACEMENT:
            return tile.player is None and tile.value == 0
        return tile.player == self._turn

    def _place_initial(self, index: int) -> None:
        self._tiles[index] = replace(self._tiles[index], value=INITIAL_ORB_VALUE, player=self._turn)
        self._turn = not self._turn
        if self._both_players_present():
            self._phase = Phase.PLAYING

    def _apply_click(self, index: int) -> None:
        tile = self._tiles[index]
        self._tiles[index] = Tile(value=tile.value + 1, player=self._turn)

    def _scan_unstable(self) -> list[int]:
        return [i for i, t in enumerate(self._tiles) if t.value >= 4]

    def _progress_step(self) -> bool:
        """Apply one wave of propagation. Returns True if anything exploded."""
        unstable = self._scan_unstable()
        if not unstable:
            return False

        for idx in unstable:
            tile = self._tiles[idx]
            owner = tile.player
            if owner is None:
                continue

            magnitude = min(tile.value - 3, 4)
            for n_idx in self._neighbors(idx):
                n = self._tiles[n_idx]
                self._tiles[n_idx] = Tile(value=n.value + magnitude, player=owner)
            self._tiles[idx] = Tile()
        return True

    def _run_chain_reaction(self) -> None:
        """Resolve waves until the board stabilizes or oscillates.

        We keep the last two snapshots; if the current board matches the
        one from two steps ago, the reaction is oscillating and we stop.
        """
        history: list[tuple[Tile, ...]] = []

        while True:
            current = tuple(self._tiles)
            if len(history) == 2 and history[0] == current:
                return
            history.append(current)
            if len(history) > 2:
                history.pop(0)

            if not self._progress_step():
                return

            if self._sole_survivor() is not None:
                return

    def _both_players_present(self) -> bool:
        has_p1 = has_p2 = False
        for t in self._tiles:
            if t.player is True:
                has_p1 = True
            elif t.player is False:
                has_p2 = True
            if has_p1 and has_p2:
                return True
        return False

    def _sole_survivor(self) -> Player:
        """True/False if only one player has orbs, None otherwise."""
        has_p1 = has_p2 = False
        for t in self._tiles:
            if t.player is True:
                has_p1 = True
            elif t.player is False:
                has_p2 = True
        if has_p1 and not has_p2:
            return True
        if has_p2 and not has_p1:
            return False
        return None

    def _finalize_turn(self) -> None:
        winner = self._sole_survivor()
        # During placement both players haven't played yet, so ignore that phase.
        if winner is not None and self._phase is Phase.PLAYING:
            self._phase = Phase.ENDED
            self._winner = winner
            return
        self._turn = not self._turn

    def _snapshot(self, *, accepted: bool) -> GameState:
        return GameState(
            tiles=tuple(self._tiles),
            size=self._size,
            turn=self._turn,
            phase=self._phase,
            winner=self._winner,
            accepted=accepted,
        )


__all__ = ["Game", "GameState", "Tile", "Phase", "BOARD_SIZE", "INITIAL_ORB_VALUE"]
