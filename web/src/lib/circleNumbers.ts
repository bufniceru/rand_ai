import { NUMBER_BALL_RADIUS } from "./possibleDrawNumberControls";

export const CIRCLE_CANVAS_SIZE = 580;
export const CIRCLE_RADIUS = 260;
export { NUMBER_BALL_RADIUS };

export const circleNumbers = Array.from({ length: 49 }, (_, index) => {
  const angle = -Math.PI / 2 + index * 2 * Math.PI / 49;
  return {
    number: index + 1,
    x: CIRCLE_CANVAS_SIZE / 2 + CIRCLE_RADIUS * Math.cos(angle),
    y: CIRCLE_CANVAS_SIZE / 2 + CIRCLE_RADIUS * Math.sin(angle),
  };
});
