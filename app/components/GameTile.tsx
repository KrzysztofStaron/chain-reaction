import React from 'react';
import { TileState } from './types';

const GameTile = ({ id, tileState }: { id: number; tileState: TileState }) => {
  return (
    <div
      id={`tile_${id}`}
      data-tile-id={id}
      className="relative bg-white/90 rounded-[14px] aspect-square shadow-md size-15 hover:bg-white/70 cursor-pointer"
    >
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
