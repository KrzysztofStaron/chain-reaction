import { TileState, TileStateFactory } from "./types";

export const createInitialBoard = (size: number): TileState[] =>
    Array.from({ length: size * size }, () => TileStateFactory());