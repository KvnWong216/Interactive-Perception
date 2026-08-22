"""Minimal candidate-conditioned calibrated interaction pipeline.

The legacy heuristic PIU implementation remains under
``interaction_uncertainty`` and is pinned by ``baselines/heuristic_v0``.
This package contains the learned reference method only.
"""

from .calibration import BinaryEffectCalibration, LACCalibrator, TemperatureScaler
from .capabilities import CapabilityRegistry
from .contracts import CandidateAction, EffectFactor, Primitive
from .controller import CalibratedSelector, ClosedLoopController

__all__ = [
    "BinaryEffectCalibration",
    "CalibratedSelector",
    "CandidateAction",
    "CapabilityRegistry",
    "ClosedLoopController",
    "EffectFactor",
    "LACCalibrator",
    "Primitive",
    "TemperatureScaler",
]
