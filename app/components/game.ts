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

  editTile(index: number, fn: (tile: TileState) => TileState): Board {
    const next: TileState[] = this.tiles.map((tile, idx) =>
      idx === index ? fn(tile) : tile,
    );
    return new Board(next, this.size);
  }

  scan(): number[] | null {
    const unstable = this.tiles
      .map((tile, idx) => (tile.value === 4 ? idx : -1))
      .filter((idx) => idx !== -1);
    return unstable.length > 0 ? unstable : null;
  }

  progress(): Board {
    // Find all unstable tiles
    const crazyTiles = this.scan();
    if (!crazyTiles) {
      return this;
    }

    let nextBoard: Board = this;

    crazyTiles.forEach((crazyIdx) => {
      const owner = nextBoard.tiles[crazyIdx].player;
      if (owner === null) {
        // skip: don't propagate from empty tile
        return;
      }

      // First, increment each neighbor by 1 and set their owner
      nextBoard.neighbors(crazyIdx).forEach((idx) => {
        nextBoard = nextBoard.editTile(idx, (tile) => ({
          ...tile,
          value: tile.value + 1,
          player: owner,
        }));
      });

      // Then, clear the source tile (reset to default)
      nextBoard = nextBoard.editTile(crazyIdx, () => TileStateFactory());
    });
    
    // If new unstable tiles were created in the process then run again
    if (nextBoard.scan()) {
      nextBoard = nextBoard.progress();
    }
    
    return nextBoard;
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

      // playing phase

      const afterMove = board.increment([index], turn).progress();
      setBoard(afterMove);
      setTurn(!turn);
    },
  };
};
