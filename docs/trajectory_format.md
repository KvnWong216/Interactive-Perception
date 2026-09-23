# Real trajectory interface

The loader accepts existing trajectories; it does not create demonstrations,
invent task success, or transform old candidate outcomes into new training data.

A JSON manifest contains `schema: "predictive-vla-trajectories-v1"` and an
`episodes` list. Each entry has exactly these required fields:

| Field | Meaning |
| --- | --- |
| `episode_id` | Unique audit identity, never a policy feature |
| `reset_family` | Correlated scenes/resets stay in one split |
| `split` | `train`, `validation`, or `test` |
| `task` | Public task language passed to the policy |
| `path` | NPZ path, relative to the manifest or absolute |
| `final_response` (optional) | Annotated `DONE: <answer>` at the final observed frame |

The NPZ is loaded with `allow_pickle=False`. For T recorded applied actions:

| Array | Shape / convention |
| --- | --- |
| `agent_rgb`, `wrist_rgb` | `[T+1,H,W,3]`, uint8; ordered cameras |
| `states` | `[T+1,8]`: eef XYZ, axis-angle, two gripper positions |
| `actions` | `[T,7]`, real applied relative LIBERO controls, before model normalization |
| `agent_depth`, `wrist_depth` (optional) | `[T+1,H,W]`, metric axial depth in the RGB pixel coordinates |
| `agent_K`, `wrist_K` (with depth) | `[3,3]` or `[T+1,3,3]`, calibrated pinhole intrinsics |
| `agent_T_world`, `wrist_T_world` (with depth) | `[T+1,4,4]`, camera-to-common-reference transforms |

Each camera supplies all three geometry arrays together or none. Invalid depth
may be NaN/zero and is marked unobserved; other inputs must be finite. RGB-only
trajectories can support replay but do not supply geometric evidence. The
dataset check reports how many episodes contain dual-camera geometry.

An action at t maps observation t to observation t+1. Do not store planned but
unexecuted tails as applied controls. Histories use the configured execution
cadence; future labels span `prediction_steps`, clipped only at the true end of
the recorded trajectory. Validation uses different reset families from training.

Raster orientation must be settled before loading. If RGB/depth are rotated,
apply the same pixel transform to K. Camera poses must be recorded at the matching
time, especially for a moving wrist camera. No hidden object geometry or simulator
success flags belong in these arrays.

The current format requires robot action trajectories. Action-free web video is
not silently treated as an action-conditioned sample. Completion annotations are
optional for replay; without valid positive examples, stop/answer learning is
not established.
