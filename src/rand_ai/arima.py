"""Bounded positional ARIMA(5,1,0), using conditional least squares."""

from collections import deque
from collections.abc import Collection

import numpy as np


class ArimaModel:
    """Forecast six sorted positions using only observed draws.

    With q=0, fitting AR(5) to first differences is a conditional
    least-squares ARIMA(5,1,0) estimate, without drift. This intentionally
    differs from the reference project's statsmodels likelihood estimator.
    """

    def __init__(self) -> None:
        self.history: deque[tuple[int, ...]] = deque(maxlen=250)

    def observe(self, drawn: Collection[int]) -> None:
        """Append a completed 6/49 draw in chronological order."""
        values = tuple(sorted(drawn))
        if len(values) != 6 or len(set(values)) != 6 or any(
            not isinstance(value, int) or not 1 <= value <= 49 for value in values
        ):
            raise ValueError("ARIMA requires six unique integers from 1 through 49")
        self.history.append(values)

    def predict(self) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        """Return relative ranking scores, not calibrated hit probabilities."""
        fallback = len(self.history) < 24
        forecasts = np.arange(1, 7, dtype=float) * 50 / 7
        scales = np.full(6, 4.0)
        if self.history:
            levels = np.asarray(self.history, dtype=float)
            forecasts = levels.mean(axis=0)
            if not fallback:
                differences = np.diff(levels, axis=0)
                # Each row contains the preceding five differences, newest first.
                features = np.stack([
                    differences[5 - lag:len(differences) - lag]
                    for lag in range(1, 6)
                ], axis=2)
                forecasts = levels[-1].copy()
                for position in range(6):
                    x = features[:, position, :]
                    y = differences[5:, position]
                    try:
                        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
                        estimate = levels[-1, position] + (
                            differences[-5:, position][::-1] @ coefficients
                        )
                        residuals = y - x @ coefficients
                        if not np.isfinite(estimate):
                            raise ValueError("Nonfinite ARIMA forecast")
                        forecasts[position] = estimate
                        scales[position] = max(float(np.sqrt(np.mean(residuals**2))), 1.0)
                    except (np.linalg.LinAlgError, ValueError):
                        forecasts[position] = levels[:, position].mean()
                        fallback = True
        # A sorted position has a narrower feasible range than an individual ball.
        forecasts = np.clip(forecasts, np.arange(1, 7), np.arange(44, 50))
        numbers = np.arange(1, 50, dtype=float)
        kernels = np.exp(-0.5 * ((numbers[:, None] - forecasts) / scales) ** 2)
        kernels /= kernels.sum(axis=0)
        scores = dict(zip(range(1, 50), (kernels.sum(axis=1) / 6).tolist()))
        status = "Positional mean fallback" if fallback else "Conditional least-squares fit"
        details = {
            number: (
                "ARIMA(5,1,0); relative ranking score, not hit probability",
                f"{status}; training draws {len(self.history)} (window 250)",
                "Position forecasts: " + ", ".join(f"{value:.2f}" for value in forecasts),
            )
            for number in scores
        }
        return scores, details
