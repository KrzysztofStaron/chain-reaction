"use client";

import { type RefObject, useLayoutEffect, useState } from "react";
import { motion } from "motion/react";
import { useAtom } from "jotai";
import { explosionsAtom, PROGRESS_STEP_MS } from "./game";

type FlyingOrb = {
  key: string;
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  owner: boolean;
};

const ORB_SIZE = 24;

const getTileCenter = (
  index: number,
  container: HTMLElement,
): { x: number; y: number } | null => {
  const tile = container.querySelector<HTMLElement>(`[data-tile-id="${index}"]`);
  if (!tile) return null;
  const boardRect = container.getBoundingClientRect();
  const rect = tile.getBoundingClientRect();
  return {
    x: rect.left - boardRect.left + rect.width / 2,
    y: rect.top - boardRect.top + rect.height / 2,
  };
};

const ExplosionOverlay = ({
  boardRef,
}: {
  boardRef: RefObject<HTMLDivElement | null>;
}) => {
  const [explosions] = useAtom(explosionsAtom);
  const [orbs, setOrbs] = useState<FlyingOrb[]>([]);

  useLayoutEffect(() => {
    const container = boardRef.current;
    if (!container || explosions.length === 0) {
      setOrbs([]);
      return;
    }

    const next: FlyingOrb[] = [];
    explosions.forEach((e, ei) => {
      const from = getTileCenter(e.from, container);
      if (!from) return;
      e.targets.forEach((target, ti) => {
        const to = getTileCenter(target, container);
        if (!to) return;
        next.push({
          key: `${ei}-${e.from}-${target}-${ti}`,
          fromX: from.x,
          fromY: from.y,
          toX: to.x,
          toY: to.y,
          owner: e.owner,
        });
      });
    });
    setOrbs(next);
  }, [explosions, boardRef]);

  if (orbs.length === 0) return null;

  return (
    <div className="pointer-events-none absolute inset-0 overflow-visible">
      {orbs.map((orb) => (
        <motion.div
          key={orb.key}
          className={`absolute rounded-full shadow-lg ${
            orb.owner ? "bg-player-1-orb" : "bg-player-2-orb"
          }`}
          style={{
            width: ORB_SIZE,
            height: ORB_SIZE,
            left: -ORB_SIZE / 2,
            top: -ORB_SIZE / 2,
          }}
          initial={{ x: orb.fromX, y: orb.fromY, scale: 0.3, opacity: 0 }}
          animate={{
            x: orb.toX,
            y: orb.toY,
            scale: [0.3, 1, 0.9],
            opacity: [0, 1, 1],
          }}
          transition={{
            duration: PROGRESS_STEP_MS / 1000,
            ease: "easeOut",
            times: [0, 0.4, 1],
          }}
        />
      ))}
    </div>
  );
};

export default ExplosionOverlay;
