"use client";

import { type MouseEvent } from "react";
import GameTile from "./components/GameTile";
import {
  gamePhaseAtom,
  turnAtom,
  useGame,
  winnerAtom,
} from "./components/game";
import { useAtom } from "jotai";

const GameBoard = () => {
  const { board, clickTile } = useGame();

  const handleTileClick = (e: MouseEvent<HTMLDivElement>, leftClick: boolean) => {
    const target = (e.target as HTMLElement).closest<HTMLElement>("[data-tile-id]");
    if (!target) return;
    clickTile(Number(target.dataset.tileId), leftClick);
  };

  const onBoardClick = (e: MouseEvent<HTMLDivElement>) => {
    handleTileClick(e, true);
  };

  const onBoardContextMenu = (e: MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    handleTileClick(e, false);
  };

  return (
    <div
      className="game grid"
      onClick={onBoardClick}
      onContextMenu={onBoardContextMenu}
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
  const [phase] = useAtom(gamePhaseAtom);
  const [winner] = useAtom(winnerAtom);

  return (
    <div className="container w-screen h-screen flex flex-col items-center justify-center gap-4" style={{ backgroundColor: "#DE7356" }}>
      <div className="flex flex-col items-center gap-1">
        {phase === "ended" && winner !== null ? (
          <div
            className={`font-semibold text-xl px-6 py-3 rounded-full shadow-md border-2 ${
              winner
                ? "bg-player-1/90 text-player-1-fg border-player-1-border/80"
                : "bg-player-2/90 text-player-2-fg border-player-2-border/80"
            }`}
          >
            Player {winner ? 1 : 2} wins
          </div>
        ) : (
          <div
            className={`font-semibold text-lg px-5 py-2 rounded-full shadow-md border-2 ${
              turn
                ? "bg-player-1/90 text-player-1-fg border-player-1-border/80"
                : "bg-player-2/90 text-player-2-fg border-player-2-border/80"
            }`}
          >
            Turn: Player {turn ? 1 : 2}
          </div>
        )}
        {phase === "placement" ? (
          <p className="text-white/90 text-sm text-center max-w-xs">
            Place your first orb on any empty cell (starts at 3).
          </p>
        ) : null}
      </div>
      <GameBoard />
    </div>
  );
}
