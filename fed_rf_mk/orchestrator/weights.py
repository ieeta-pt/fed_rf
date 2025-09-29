from __future__ import annotations

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class WeightManager:
    """Handles weight management, normalization, and validation for federated learning.

    Normalization semantics:
    - If all weights are None or negative for successful silos → assign equal weights.
    - If some weights are defined (>=0) and some are None/negative → distribute remaining
      mass equally among undefined ones, then renormalize strictly to sum 1.0.
    - If all defined → renormalize strictly to sum 1.0.
    """

    def __init__(self) -> None:
        self.weights: Dict[str, Optional[float]] = {}

    def set_weight(self, silo_name: str, weight: Optional[float]) -> None:
        self.weights[silo_name] = weight

    def get_weight(self, silo_name: str) -> Optional[float]:
        return self.weights.get(silo_name)

    def get_all_weights(self) -> Dict[str, Optional[float]]:
        return self.weights.copy()

    def normalize_weights(self, successful_silos: list[str]) -> Dict[str, float]:
        if not successful_silos:
            logger.warning("No successful silos. Returning empty normalized weights.")
            return {}

        # Gather raw weights for successful silos
        raw = {s: self.weights.get(s) for s in successful_silos}
        defined = {s: float(w) for s, w in raw.items() if w is not None and float(w) >= 0.0}
        undefined = [s for s in successful_silos if s not in defined]

        if not defined and undefined:
            # All unspecified -> equal share
            eq = 1.0 / len(successful_silos)
            out = {s: eq for s in successful_silos}
            logger.info("All weights were None/negative. Assigning equal weights: %s", eq)
        else:
            # Start with defined values
            sum_defined = sum(defined.values())
            out = defined.copy()
            if undefined:
                # Distribute remaining probability mass equally among undefined
                remaining = max(0.0, 1.0 - sum_defined)
                fill = remaining / len(undefined) if undefined else 0.0
                for s in undefined:
                    out[s] = fill
                logger.info("Some weights were None/negative. Distributed fill=%s to: %s", fill, undefined)

            # Strict renormalization to avoid drift and sums > 1.0
            total = sum(out.values()) or 1.0
            out = {s: v / total for s, v in out.items()}
            logger.info("Normalized weights among successful clients: %s", out)

        # Persist normalized values
        for s in successful_silos:
            self.weights[s] = out[s]

        return out

