import React from 'react';
import { TileState } from './types';

const DirectionIcon = ({ direction }: { direction: TileState['direction'] }) => (
  <svg
    aria-hidden
    className={`pointer-events-none absolute bottom-1 right-1 size-3.5 shrink-0 text-black/45 ${
      direction === 'diagonal' ? 'rotate-45' : ''
    }`}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2.25"
    strokeLinecap="round"
  >
    <path d="M12 5v14M5 12h14" />
  </svg>
);

const GameTile = ({ id, tileState }: { id: number; tileState: TileState }) => {
  return (
    <div
      id={`tile_${id}`}
      data-tile-id={id}
      className="relative bg-white/90 rounded-[14px] aspect-square shadow-md size-15 hover:bg-white/70 cursor-pointer"
    >
      <DirectionIcon direction={tileState.direction} />
      {tileState.player !== null && (
        <div
          className={`absolute inset-2 rounded-full text-black flex items-center justify-center ${
            tileState.player === true ? "bg-player-1-orb" : "bg-player-2-orb"
          }`}
        >
             {tileState.value}
        </div>
      )}

   
    </div>
  );
};

export default GameTile;
