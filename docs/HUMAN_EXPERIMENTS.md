# Decisions and data that still require collaborators

Automated plumbing is tracked by `benchmarks/rss_v1/gates.yaml`. The items
below require a scientific choice, manual annotation, or a new external model;
code should not silently decide them after seeing test results.

## 1. Freeze the paper test distribution

All T01 prompt-state and action-effect scenes collected so far are
calibration-only. Before any paper comparison, collaborators must freeze:

- held-out objects and prompts;
- held-out container layouts;
- clutter and occlusion severity bins;
- scene seeds and trial counts;
- primary and secondary metrics;
- which failures count as unsafe, incomplete, or `NOT_FOUND`.

The test set must not reuse any block in the authoritative
`benchmarks/rss_v1/seed_registry.yaml`. In particular, 500--599 is a legacy
quarantine, 600--659 has been consumed by development/diagnosis, and 700--799
is permanently debug-only. Seeds 660--699 were consumed by the v9 NOT-GO run
and later contaminated by v10 diagnosis. Seeds 1400--1439 are the frozen clean
development extension, 800--899 are later non-paper closed-loop validation, and 900--999
are the one-time sealed audit.

## 2. Choose final-task losses

The target-observability experiment reports a full information-cost sweep. A
final-task claim needs costs in a shared unit:

- execution time or energy;
- contact/collision penalty;
- failed manipulation penalty;
- false-commit penalty;
- false-`NOT_FOUND` penalty;
- human-assistance penalty, if assistance is allowed.
- temporal-logic violation penalty;
- `SAFE_STOP` / deferral cost, which must remain distinct from `NOT_FOUND`.

Freeze these values from task requirements or independent measurements. Do not
choose them from paper test success.

## 3. Freeze the prompt-to-temporal-specification contract

The runtime automaton may contain generic rules such as “commit only after
sufficient evidence” and “NOT_FOUND only after search exhaustion.” It must not
contain benchmark answers such as `T01 -> OPEN_DRAWER` or object-specific search
orders. Collaborators must freeze:

- the allowed atomic propositions;
- the generic finite-trace rules;
- how calibrated proposition sets induce a progress-state distribution;
- which violation classes are safety-critical;
- whether `SAFE_STOP` means wait, ask for help, or terminate without a task
  claim.

Do this before the paper test tasks are revealed. T-LEAF-style learned logic is
an optional sample-efficiency arm, not permission to learn scenario answers
from the test set.

## 4. Choose the post-reveal executor contract

Open-drawer butter retrieval is 0/5. Collaborators must choose one scope before
more final-task trials:

- keep pure final-goal `pi05_libero` and report the capability boundary;
- use a preregistered decomposition into training-aligned unit prompts;
- replace the executor with another frozen policy;
- train an executor and narrow the paper claim accordingly.

Whichever contract is chosen must pass its own context-specific 0.90 lower-
bound gate. Stock LIBERO object success cannot be transferred to an open drawer.

## 5. Freeze the paper-scale `EMPTY` certificate

Simulator knowledge that the target was placed elsewhere is not observable
evidence that a container is empty. T01 development uses a conservative
seed-paired target-present counterfactual only as a calibration proxy.
Collaborators must freeze one paper-scale certificate before `NOT_FOUND` tests:

- depth-based visible/occluded container volume;
- geometry-grounded visibility with independently validated RGB prediction; or
- a deliberately instrumented container protocol whose coverage transfers to
  the uninstrumented test scenes.

The chosen certificate must reject `OPENED_UNOBSERVED`: an opened drawer whose
interior is still blocked by the arm or outside both policy cameras.

## 6. Select the second frozen policy

The generality gate requires a second VLA. Freeze checkpoint, commit, camera
contract, prompt form, and supported action family before running custom
scenes. OpenVLA or Octo are candidates only after a stock reproduction gate.

## 7. Qualitative audit

Two collaborators should independently inspect a preregistered sample of
success and failure videos and label:

- correct target reference;
- correct information action;
- physical action success;
- whether new prompt-relevant information entered a policy camera;
- final-task completion;
- safety/contact failure.

Resolve disagreements without access to method identity. The simulator labels
remain useful, but they do not replace this failure-taxonomy audit.

## Required paper arms

- monolithic frozen VLA;
- prompt-blind fixed rule;
- raw action spread;
- CoMe-style binary information-sufficiency head;
- uncalibrated belief head;
- visual-only history;
- no-history outcome critic;
- no-effect risk planner;
- no temporal-progress state;
- deterministic argmax automaton;
- complete method;
- oracle information action as an upper bound.

## Claims still forbidden

- π0.5 is broadly overfit;
- the method improves final task success;
- conformal coverage guarantees physical success;
- the method generalizes across scenes or VLAs;
- `VIEWPOINT_BLOCKED`, `ABSENT`, or `NOT_FOUND` is solved;
- the paired-RGB critic is reliable before its frozen audit passes.
