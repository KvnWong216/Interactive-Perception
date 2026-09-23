"""Public control feedback shared by the isolated simulator and evaluator."""

import numpy as np


def applied_prefix(actions, low, high):
    """Record the bounded command sent to LIBERO, including binary gripper."""
    values = np.asarray(actions, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[1] != 7
        or not len(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError("invalid action prefix")
    values = np.clip(values, low, high).astype(np.float32)
    values[:, -1] = np.where(values[:, -1] < 0, -1.0, 1.0)
    return values


def validate_feedback(packet, *, previous_step, requested_count):
    """Never substitute predicted future actions for executed feedback."""
    step = packet["step"]
    rows = packet["applied_actions"]
    if type(step) is not int or step - previous_step != requested_count:
        raise ValueError("simulator step does not match requested execution prefix")
    if [row["step"] for row in rows] != list(range(previous_step, step)):
        raise ValueError("simulator feedback is missing or misaligned")
    values = np.asarray([row["value"] for row in rows], dtype=np.float32)
    if values.shape != (requested_count, 7) or not np.isfinite(values).all():
        raise ValueError("invalid executed control feedback")
    return values
