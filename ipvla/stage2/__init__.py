"""Stage-2 closed-loop active perception: cerebellum inertia, inspect memory, switching & planner."""
from ipvla.stage2.cerebellum import CerebellumConfig, ContinuousCerebellumInertia, InertiaSnapshot
from ipvla.stage2.memory import BufferedFrame, InspectKeyframeMemoryBuffer, MemoryConfig
from ipvla.stage2.switching import (
    CommitDecision,
    SwitchingConfig,
    TwoStageSwitchingController,
    WakeDecision,
    normalize_subtask,
    subtasks_equivalent,
)

__all__ = [
    "BufferedFrame",
    "CerebellumConfig",
    "CommitDecision",
    "ContinuousCerebellumInertia",
    "InertiaSnapshot",
    "InspectKeyframeMemoryBuffer",
    "MemoryConfig",
    "SwitchingConfig",
    "TwoStageSwitchingController",
    "WakeDecision",
    "normalize_subtask",
    "subtasks_equivalent",
]
