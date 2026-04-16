import React from 'react';
import { TileState } from './types';

const GameTile = ({ id, tileState }: { id: number; tileState: TileState }) => {
  return (
    <div
      id={`tile_${id}`}
      data-tile-id={id}
      className="bg-white/90 rounded-[14px] aspect-square flex items-center justify-center shadow-md size-11 hover:bg-white/70 cursor-pointer"
    >

    </div>
  );
};

export default GameTile;
