export type Direction = "orthogonal" | "diagonal"

export type TileState =
    | { player: null, value: 0, direction: Direction }
    | { player: boolean, value: number, direction: Direction }

export const TileStateFactory = (): TileState => ({
    value: 0,
    direction: "orthogonal",
    player: null,
})
