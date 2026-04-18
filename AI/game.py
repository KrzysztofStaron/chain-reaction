"""
Chain Reaction - pure-compute Python port (optimized).

The public surface of `Game` is intentionally tiny:
    - __init__(size=5)
    - click_tile(x, y) -> GameState

Internally the board is stored as two flat lists (values, players) to
avoid frozen-dataclass allocation in the chain-reaction loop. GameState
exposes the raw tuples for fast access; Tile objects are only created
on-demand via `tile_at()` or the `tiles` property.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional


BOARD_SIZE = 5
INITIAL_ORB_VALUE = 3

_N = BOARD_SIZE * BOARD_SIZE  # 25


class Phase(str, Enum):
    PLACEMENT = "placement"
    PLAYING = "playing"
    ENDED = "ended"


Player = Optional[bool]


@dataclass(frozen=True, slots=True)
class Tile:
    value: int = 0
    player: Player = None


@dataclass(frozen=True, slots=True)
class GameState:
    """Immutable snapshot returned by `click_tile`.

    The board data is stored as raw tuples for fast access:
        values[i]  — orb value at flat index i (0 if empty)
        players[i] — owner at flat index i (True/False/None)
    Tile objects are only created when you call `tile_at` or `tiles`.
    """
    values: tuple[int, ...]
    players: tuple
    size: int
    turn: bool
    phase: Phase
    winner: Player = None
    accepted: bool = True

    @property
    def tiles(self) -> tuple[Tile, ...]:
        vs, ps = self.values, self.players
        return tuple(Tile(vs[i], ps[i]) for i in range(len(vs)))

    def tile_at(self, x: int, y: int) -> Tile:
        idx = y * self.size + x
        return Tile(self.values[idx], self.players[idx])


# ---------------------------------------------------------------------------
# Precomputed neighbor table (module-level, computed once)
# ---------------------------------------------------------------------------
def _build_neighbor_table(size: int) -> tuple[tuple[int, ...], ...]:
    offsets = ((0, -1), (0, 1), (-1, 0), (1, 0))
    table: list[tuple[int, ...]] = []
    for i in range(size * size):
        x, y = i % size, i // size
        nb: list[int] = []
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                nb.append(ny * size + nx)
        table.append(tuple(nb))
    return tuple(table)


_NEIGHBORS: tuple[tuple[int, ...], ...] = _build_neighbor_table(BOARD_SIZE)


class Game:
    __slots__ = ("_size", "_values", "_players", "_turn", "_phase", "_winner")

    def __init__(self, size: int = BOARD_SIZE, first_player: Optional[bool] = None) -> None:
        self._size = size
        self._values: list[int] = [0] * _N
        self._players: list[Player] = [None] * _N
        self._turn: bool = random.random() < 0.5 if first_player is None else first_player
        self._phase: Phase = Phase.PLACEMENT
        self._winner: Player = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def click_tile(self, x: int, y: int) -> GameState:
        if not (0 <= x < self._size and 0 <= y < self._size) or self._phase is Phase.ENDED:
            return self._snapshot(False)

        idx = y * self._size + x

        if not self._can_interact(idx):
            return self._snapshot(False)

        if self._phase is Phase.PLACEMENT:
            self._place_initial(idx)
            return self._snapshot(True)

        self._apply_click(idx)
        self._run_chain_reaction()
        self._finalize_turn()
        return self._snapshot(True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _can_interact(self, idx: int) -> bool:
        if self._phase is Phase.PLACEMENT:
            return self._players[idx] is None
        return self._players[idx] == self._turn

    def _place_initial(self, idx: int) -> None:
        self._values[idx] = INITIAL_ORB_VALUE
        self._players[idx] = self._turn
        self._turn = not self._turn
        p = self._players
        has_p1 = has_p2 = False
        for i in range(_N):
            pi = p[i]
            if pi is True:
                has_p1 = True
            elif pi is False:
                has_p2 = True
            if has_p1 and has_p2:
                self._phase = Phase.PLAYING
                return

    def _apply_click(self, idx: int) -> None:
        self._values[idx] += 1

    def _run_chain_reaction(self) -> None:
        """Resolve waves until the board stabilizes or oscillates."""
        vals = self._values
        plrs = self._players

        prev2: tuple | None = None
        prev1: tuple | None = None

        while True:
            current = (*vals, *plrs)
            if prev2 is not None and prev2 == current:
                return
            prev2 = prev1
            prev1 = current

            unstable = [i for i in range(_N) if vals[i] >= 4]
            if not unstable:
                return

            for idx in unstable:
                owner = plrs[idx]
                if owner is None:
                    continue
                mag = vals[idx] - 3
                if mag > 4:
                    mag = 4
                for n in _NEIGHBORS[idx]:
                    vals[n] += mag
                    plrs[n] = owner
                vals[idx] = 0
                plrs[idx] = None

            has_p1 = has_p2 = False
            for i in range(_N):
                pi = plrs[i]
                if pi is True:
                    has_p1 = True
                elif pi is False:
                    has_p2 = True
                if has_p1 and has_p2:
                    break
            if has_p1 != has_p2:
                return

    def _finalize_turn(self) -> None:
        plrs = self._players
        has_p1 = has_p2 = False
        for i in range(_N):
            pi = plrs[i]
            if pi is True:
                has_p1 = True
            elif pi is False:
                has_p2 = True
        winner: Player = None
        if has_p1 and not has_p2:
            winner = True
        elif has_p2 and not has_p1:
            winner = False
        if winner is not None and self._phase is Phase.PLAYING:
            self._phase = Phase.ENDED
            self._winner = winner
            return
        self._turn = not self._turn

    def _snapshot(self, accepted: bool) -> GameState:
        return GameState(
            values=tuple(self._values),
            players=tuple(self._players),
            size=self._size,
            turn=self._turn,
            phase=self._phase,
            winner=self._winner,
            accepted=accepted,
        )


__all__ = ["Game", "GameState", "Tile", "Phase", "BOARD_SIZE", "INITIAL_ORB_VALUE"]

