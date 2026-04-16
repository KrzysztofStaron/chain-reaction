import React from 'react';
import { TileState } from './types';

const DirectionIcon = ({ direction }: { direction: TileState['direction'] }) => (
  <svg
    aria-hidden
    className={`pointer-events-none absolute inset-0 w-full h-full text-black/20 ${
      direction === 'diagonal' ? 'rotate-45' : ''
    }`}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.5"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M12 7V3M9 6l3-3 3 3" />
    <path d="M12 17v4M9 18l3 3 3-3" />
    <path d="M7 12H3M6 9l-3 3 3 3" />
    <path d="M17 12h4M18 9l3 3-3 3" />
  </svg>
);

const GameTile = ({ id, tileState }: { id: number; tileState: TileState }) => {
  const bgClass =
    tileState.player === null
      ? 'bg-white/90 hover:bg-white/70'
      : tileState.player
        ? 'bg-player-1-orb hover:bg-player-1-orb/80'
        : 'bg-player-2-orb hover:bg-player-2-orb/80';

  const fgClass = tileState.player === null ? 'text-black' : 'text-white';

  return (
    <div
      id={`tile_${id}`}
      data-tile-id={id}
      className={`relative rounded-[14px] aspect-square shadow-md size-15 cursor-pointer overflow-hidden flex items-center justify-center transition-colors ${bgClass}`}
    >
      {tileState.value > 0 && (
        <>
          <DirectionIcon direction={tileState.direction} />
          <span className={`text-lg font-semibold ${fgClass}`}>
            {tileState.value}
          </span>
        </>
      )}
    </div>
  );
}

export default GameTile;
