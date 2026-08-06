"""Interactive Perception: measuring what a VLA does when it cannot see enough.

The scenario definitions in ``scenarios/`` put a target where a fixed camera
cannot resolve it and no camera pose would help.  This package drives a policy
through those scenes and records, at every step, both what the policy did and
how much evidence its action distribution actually carried.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.3.0"
