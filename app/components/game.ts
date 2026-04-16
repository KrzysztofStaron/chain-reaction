import { atom, useAtom } from "jotai";
import { Direction, TileState, TileStateFactory } from "./types";

export const BOARD_SIZE = 5;

/** Value of each player's first orb when placed during the opening phase */
export const INITIAL_ORB_VALUE = 3;

export type GamePhase = "placement" | "playing";

export class Board {
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

  canInteract(index: number, turn: boolean, phase: GamePhase): boolean {
    const tile = this.tiles[index];
    if (phase === "placement") {
      return tile.player === null && tile.value === 0;
    }
    return tile.player === turn;
  }

  placeInitial(index: number, player: boolean): Board {
    const next: TileState[] = this.tiles.map((tile, idx) =>
      idx === index
        ? { direction: tile.direction, player, value: INITIAL_ORB_VALUE }
        : tile,
    );
    return new Board(next, this.size);
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

  neighbors(index: number): number[] {
    const inBounds = (x: number, y: number): boolean =>
      x >= 0 && x < this.size && y >= 0 && y < this.size;

    const [x, y] = this.toXY(index);
    const offsets = Board.OFFSETS[this.tiles[index].direction];
    return offsets
      .map(([dx, dy]) => [x + dx, y + dy] as const)
      .filter(([nx, ny]) => inBounds(nx, ny))
      .map(([nx, ny]) => this.toIndex(nx, ny));
  }

  increment(indices: number[], player: boolean): Board {
    const set = new Set(indices);
    const next: TileState[] = this.tiles.map((tile, idx) =>
      set.has(idx)
        ? { direction: tile.direction, value: tile.value + 1, player }
        : tile,
    );
    return new Board(next, this.size);
  }

  progress(): Board {
    // placeholder for future explosion / chain rules
    return this;
  }
}

export const gameStateAtom = atom<Board>(Board.create(BOARD_SIZE));
export const turnAtom = atom(true);
export const gamePhaseAtom = atom<GamePhase>("placement");

export const useGame = () => {
  const [board, setBoard] = useAtom(gameStateAtom);
  const [turn, setTurn] = useAtom(turnAtom);
  const [phase, setPhase] = useAtom(gamePhaseAtom);
  return {
    board,
    turn,
    phase,
    clickTile: (index: number) => {
      if (!board.canInteract(index, turn, phase)) return;

      if (phase === "placement") {
        const nextBoard = board.placeInitial(index, turn);
        setBoard(nextBoard);
        setTurn(!turn);
        if (Board.bothPlayersPresent(nextBoard.tiles)) {
          setPhase("playing");
          setTurn(true);
        }
        return;
      }

      setBoard(board.increment([index], turn));
      setTurn(!turn);
    },
  };
};
