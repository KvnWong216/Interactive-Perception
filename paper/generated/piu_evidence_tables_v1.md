# PIU evidence tables

> Generated from hash-checked artifacts. `PENDING` means no admissible artifact exists; it never means zero. Development results are not sealed test evidence. B6/B7 are oracle upper bounds.

## Evidence readiness

| artifact | status | SHA-256 |
|---|---:|---|
| retained_negative | AVAILABLE | 25e666cfcd23671957c99ba2157e98e65365e9667d6827ad8de4814b7bcb0eed |
| binding_development | PENDING | PENDING |
| effect_development | PENDING | PENDING |
| target_binding_sealed | PENDING | PENDING |
| action_effect_sealed | PENDING | PENDING |
| closed_loop_sealed | PENDING | PENDING |

## Retained fixed-scenario negative/qualification evidence

| condition | metric | result |
|---|---|---:|
| direct_closed_butter | target_pick | 0/10 (0.0%) |
| direct_closed_butter | wrong_object_contact | 9/10 (90.0%) |
| open_closed_drawer | drawer_open | 9/10 (90.0%) |
| open_closed_drawer | information_acquired | 8/10 (80.0%) |
| direct_after_actual_open | target_visible_initial | 8/10 (80.0%) |
| direct_after_actual_open | target_pick | 0/10 (0.0%) |
| direct_after_actual_open | wrong_object_contact | 3/10 (30.0%) |
| direct_after_actual_open | task_success | 0/10 (0.0%) |
| direct_visible_cream_cheese | target_pick | 10/10 (100.0%) |
| direct_visible_cream_cheese | target_destination_final | 3/10 (30.0%) |

## Binding input ablations (development only)

| ablation | spatial_nll | point_hit | target_probability_mass | presence_brier | sufficiency_brier | holding_brier |
|---|---:|---:|---:|---:|---:|---:|
| full | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| no_prompt | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| prompt_swap | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| last_frame_only | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| mean_image_tokens | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| no_action_history | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| agent_view_only | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| wrist_view_only | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| shuffled_spatial_positions | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| shuffled_temporal_order | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## Effect/route variants (development only)

| variant | route_nll | route_top1_accuracy | macro_supported_factor_brier |
|---|---:|---:|---:|
| route_only | PENDING | PENDING | PENDING |
| stop_gradient_effect | PENDING | PENDING | PENDING |
| joint_effect | PENDING | PENDING | PENDING |

## Sealed paired B0--B8 table

| method | class | target_grasp_contact | wrong_object_grasp_contact | target_destination_final | task_success | abstention | interaction_count | executed_steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | public_method | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| B1 | public_method | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| B2 | public_method | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| B3 | public_method | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| B4 | public_method | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| B5 | public_method | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| B6 | oracle_upper_bound | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| B7 | oracle_upper_bound | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| B8 | public_method | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

Continuous cells are `mean / median`. This table contains no automatic success decision.

## Claim boundary

- Main-table evidence complete: `false`.
- Automatic method success: `null`.
- Broad OOD claim allowed: `false`.
- Missing values encoded as zero: `false`.
