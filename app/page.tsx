"use client";

import { type MouseEvent } from "react";
import GameTile from "./components/GameTile";
import { turnAtom, useGame } from "./components/game";
import { useAtom } from "jotai";

const GameBoard = () => {
  const { board, clickTile } = useGame();

  const onBoardClick = (e: MouseEvent<HTMLDivElement>) => {
    const target = (e.target as HTMLElement).closest<HTMLElement>("[data-tile-id]");
    if (!target) return;
    clickTile(Number(target.dataset.tileId));
  };

  return (
    <div
      className="game grid"
      onClick={onBoardClick}
      style={{
        gridTemplateColumns: `repeat(${board.size}, minmax(0, 1fr))`,
        gridTemplateRows: `repeat(${board.size}, minmax(0, 1fr))`,
        gap: "0.25rem",
      }}
    >
      {board.tiles.map((tileState, idx) => (
        <GameTile key={idx} id={idx} tileState={tileState} />
      ))}
    </div>
  );
};

export default function Home() {
  const [turn] = useAtom(turnAtom);

  return (
    <div className="container w-screen h-screen flex flex-col items-center justify-center gap-4" style={{ backgroundColor: "#DE7356" }}>
      <div
        className={`font-semibold text-lg px-5 py-2 rounded-full shadow-md border-2 ${
          turn
            ? "bg-player-1/90 text-player-1-fg border-player-1-border/80"
            : "bg-player-2/90 text-player-2-fg border-player-2-border/80"
        }`}
      >
        Turn: Player {turn ? 1 : 2}
      </div>
      <GameBoard />
    </div>
  );
}
