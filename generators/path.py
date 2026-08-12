"""Shared output contract every generator arm returns, so the
stylized-facts/conformal harness (benchmark/stylized_facts.py,
benchmark/conformal.py, benchmark/generator_ladder.py) never needs to
know which generator produced a path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GeneratedPath:
    generator_id: str
    log_returns: np.ndarray
    seed: int
    params: dict = field(default_factory=dict)
