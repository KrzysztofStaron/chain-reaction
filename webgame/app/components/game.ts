import { atom, useAtom } from "jotai";
import { useState } from "react";
import { Direction, TileState, TileStateFactory } from "./types";

export const BOARD_SIZE = 5;

/** Value of each player's first orb when placed during the opening phase */
export const INITIAL_ORB_VALUE = 3;

/** Duration of a single propagation wave animation, ms */
export const PROGRESS_STEP_MS = 500;

export type GamePhase = "placement" | "playing" | "ended";

export type Explosion = {
  from: number;
  owner: boolean;
  targets: number[];
};

export class Board {
  /** Per-direction neighbor coordinate offsets */
  private static readonly OFFSETS: Record<
    Direction,
    ReadonlyArray<readonly [number, number]>
  > = {
    orthogonal: [[0, -1], [0, 1], [-1, 0], [1, 0]],
    diagonal: [[-1, -1], [1, -1], [-1, 1], [1, 1]],
  };

  constructor(
    readonly tiles: readonly TileState[],
    readonly size: number,
  ) {}

  /** Create an empty board */
  static create(size: number): Board {
    return new Board(
      Array.from({ length: size * size }, () => TileStateFactory()),
      size,
    );
  }

  toXY(index: number): readonly [number, number] {
    return [index % this.size, Math.floor(index / this.size)];
  }

  toIndex(x: number, y: number): number {
    return y * this.size + x;
  }

  private withDraft(mutator: (draft: TileState[]) => void): Board {
    const draft = this.tiles.slice();
    mutator(draft);
    return new Board(draft, this.size);
  }

  private neighborsForDirection(index: number, direction: Direction): number[] {
    const [x, y] = this.toXY(index);
    const offsets = Board.OFFSETS[direction];
    const result: number[] = [];

    for (const [dx, dy] of offsets) {
      const nx = x + dx;
      const ny = y + dy;
      if (nx >= 0 && nx < this.size && ny >= 0 && ny < this.size) {
        result.push(this.toIndex(nx, ny));
      }
    }

    return result;
  }

  neighbors(index: number): number[] {
    return this.neighborsForDirection(index, this.tiles[index].direction);
  }

  canInteract(index: number, turn: boolean, phase: GamePhase): boolean {
    if (phase === "ended") return false;
    const tile = this.tiles[index];
    if (phase === "placement") {
      return tile.player === null && tile.value === 0;
    }
    return tile.player === turn;
  }

  placeInitial(index: number, player: boolean): Board {
    const tile = this.tiles[index];
    return this.withDraft((draft) => {
      draft[index] = {
        direction: tile.direction,
        player,
        value: INITIAL_ORB_VALUE,
      };
    });
  }

  static tilesEqual(
    a: readonly TileState[],
    b: readonly TileState[],
  ): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (
        a[i].value !== b[i].value ||
        a[i].player !== b[i].player ||
        a[i].direction !== b[i].direction
      ) {
        return false;
      }
    }
    return true;
  }

  static bothPlayersPresent(tiles: readonly TileState[]): boolean {
    let hasP1 = false;
    let hasP2 = false;
    for (const t of tiles) {
      if (t.player === true) hasP1 = true;
      if (t.player === false) hasP2 = true;
      if (hasP1 && hasP2) return true;
    }
    return false;
  }

  /** `true` = only player 1 has orbs, `false` = only player 2, `null` = both or neither still present */
  static soleSurvivor(tiles: readonly TileState[]): boolean | null {
    let hasP1 = false;
    let hasP2 = false;
    for (const t of tiles) {
      if (t.player === true) hasP1 = true;
      if (t.player === false) hasP2 = true;
    }
    if (hasP1 && hasP2) return null;
    if (hasP1 && !hasP2) return true;
    if (!hasP1 && hasP2) return false;
    return null;
  }

  increment(indices: number[], player: boolean): Board {
    const set = new Set(indices);
    if (set.size === 0) return this;

    return this.withDraft((draft) => {
      for (const idx of set) {
        const tile = draft[idx];
        draft[idx] = {
          direction: tile.direction,
          value: tile.value + 1,
          player,
        };
      }
    });
  }

  editTile(index: number, fn: (tile: TileState) => TileState): Board {
    const updated = fn(this.tiles[index]);
    if (updated === this.tiles[index]) return this;

    return this.withDraft((draft) => {
      draft[index] = updated;
    });
  }

  scan(): number[] | null {
    const unstable: number[] = [];
    for (let idx = 0; idx < this.tiles.length; idx++) {
      if (this.tiles[idx].value >= 4) unstable.push(idx);
    }
    return unstable.length > 0 ? unstable : null;
  }

  /**
   * Applies a single wave of propagation: every currently-unstable tile
   * splits once, pushing `value - 3` orbs (capped at 4) to each neighbor.
   * Returns the resulting board along with the explosions that occurred
   * (for animation).
   */
  progressStep(): { board: Board; explosions: Explosion[] } {
    const unstableTiles = this.scan();
    if (!unstableTiles) {
      return { board: this, explosions: [] };
    }

    const nextTiles = this.tiles.slice();
    const explosions: Explosion[] = [];

    unstableTiles.forEach((unstableIdx) => {
      const tile = nextTiles[unstableIdx];
      const owner = tile.player;
      if (owner === null) return;

      const magnitude = Math.min(tile.value - 3, 4);
      const neighbors = this.neighborsForDirection(unstableIdx, tile.direction);
      const targets = neighbors.flatMap((n) =>
        Array<number>(magnitude).fill(n),
      );
      explosions.push({ from: unstableIdx, owner, targets });

      neighbors.forEach((idx) => {
        const neighbor = nextTiles[idx];
        nextTiles[idx] = {
          direction: neighbor.direction,
          value: neighbor.value + magnitude,
          player: owner,
        };
      });

      nextTiles[unstableIdx] = TileStateFactory();
    });

    return { board: new Board(nextTiles, this.size), explosions };
  }
}

export const turnAtom = atom(Math.random() < 0.5);
export const gamePhaseAtom = atom<GamePhase>("placement");
/** Set when `phase === "ended"`: `true` = player 1 won, `false` = player 2 */
export const winnerAtom = atom<boolean | null>(null);
/** Explosions currently animating on the board. Empty when idle. */
export const explosionsAtom = atom<Explosion[]>([]);

export const useGame = () => {
  const [board, setBoard] = useState<Board>(() => Board.create(BOARD_SIZE));
  const [animating, setAnimating] = useState(false);
  const [turn, setTurn] = useAtom(turnAtom);
  const [phase, setPhase] = useAtom(gamePhaseAtom);
  const [, setWinner] = useAtom(winnerAtom);
  const [, setExplosions] = useAtom(explosionsAtom);

  const runChainReaction = (startBoard: Board, nextTurn: boolean) => {
    setAnimating(true);

    const history: Board[] = [];

    const resolve = (final: Board) => {
      setExplosions([]);
      setBoard(final);
      setAnimating(false);

      const sole = Board.soleSurvivor(final.tiles);
      if (sole !== null) {
        setPhase("ended");
        setWinner(sole);
        return;
      }
      setTurn(nextTurn);
    };

    const step = (current: Board) => {
      const twoAgo = history[0];
      if (twoAgo && Board.tilesEqual(twoAgo.tiles, current.tiles)) {
        resolve(current);
        return;
      }
      history.push(current);
      if (history.length > 2) history.shift();

      const { board: next, explosions } = current.progressStep();

      if (explosions.length === 0) {
        resolve(current);
        return;
      }

      setBoard(current);
      setExplosions(explosions);

      setTimeout(() => {
        setExplosions([]);
        setBoard(next);
        step(next);
      }, PROGRESS_STEP_MS);
    };

    step(startBoard);
  };

  return {
    board,
    turn,
    phase,
    animating,
    clickTile: (index: number, leftClick: boolean) => {
      if (phase === "ended") return;
      if (animating) return;
      if (!board.canInteract(index, turn, phase)) return;

      if (phase === "placement") {
        const nextBoard = board.placeInitial(index, turn);
        setBoard(nextBoard);
        setTurn(!turn);

        if (Board.bothPlayersPresent(nextBoard.tiles)) {
          setPhase("playing");
        }
        return;
      }

      const afterMove = board.editTile(index, (tile) => {
        const nextDirection = leftClick
          ? tile.direction
          : tile.direction === "orthogonal"
            ? "diagonal"
            : "orthogonal";

        return {
          direction: nextDirection,
          value: tile.value + 1,
          player: turn,
        };
      });

      runChainReaction(afterMove, !turn);
    },
  };
};
