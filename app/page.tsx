"use client";

import { type MouseEvent } from "react";
import GameTile from "./components/GameTile";
import { atom, useAtom } from 'jotai';
import { TileState, TileStateFactory } from "./components/types";
import { createInitialBoard } from "./components/utils";

const BOARDSIZE = 5;
const gameStateAtom = atom<TileState[]>(createInitialBoard(BOARDSIZE));

const GameBoard = ({size} : {size: number}) => {
  const [gameState] = useAtom(gameStateAtom);

  const onBoardClick = (e: MouseEvent<HTMLDivElement>) => {
    const target = (e.target as HTMLElement).closest<HTMLElement>("[data-tile-id]");
    if (!target) return;
    const tileId = Number(target.dataset.tileId);
    console.log(`Tile clicked: ${tileId}`);
  };

  return (
    <div
      className="game grid"
      onClick={onBoardClick}
      style={{
        gridTemplateColumns: `repeat(${size}, minmax(0, 1fr))`,
        gridTemplateRows: `repeat(${size}, minmax(0, 1fr))`,
        gap: "0.25rem",
      }}
    >
      {gameState.map((tileState, idx) => (
        <GameTile key={idx} id={idx} tileState={tileState} />
      ))}
    </div>
  );
}

export default function Home() {
  return (
    <div className="container w-screen h-screen flex items-center justify-center" style={{ backgroundColor: "#DE7356" }}>
      <GameBoard size={BOARDSIZE}/>
    </div>
  );
}
