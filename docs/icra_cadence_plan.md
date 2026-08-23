# ICRA-cadence research plan (internal only)

This project is not being submitted to ICRA. The official ICRA 2027 paper
deadline, **2026-09-15 23:59 Pacific**, is used only as an internal forcing
function. The official dates are maintained by the
[ICRA 2027 call for papers](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/).

## Operating rules

- Every week ends with an immutable artifact, a failing/passing gate, and a
  short decision memo. “Still tuning” is not a deliverable.
- Keep the original T01D drawer BDDL fixed during the first cycle. Vary seeds,
  prompt target, drawer state, candidate branch, and policy sampling only.
- Group every same-state counterfactual by seed. Development selects models;
  calibration fits temperature/conformal thresholds; test is opened once.
- Report route quality, physical information acquisition, target recognition,
  final manipulation, and task success separately.
- Stop a branch after two controlled iterations fail to improve its prespecified
  metric. Do not accumulate prompt/rule patches.

## Calendar and gates

| dates (Pacific) | deliverable | blocking gate |
|---|---|---|
| Aug 22-24 | environment audit, frozen protocol, B0 direct and oracle-route executor qualification | exact commands reproduce; no online privileged input |
| Aug 25-28 | same-state prompt counterfactual collection and shared-VLM feature cache | hashes valid; paired seeds never cross splits |
| Aug 29-Sep 2 | route-only B6 and effect+route B7 smoke training | overfit a tiny batch; development loss beats uniform; no leakage |
| Sep 3-6 | temperature/effect calibration and abstention evaluation | empirical coverage reported with set size; zero forced top-1 |
| Sep 7-10 | main B0/B2/B6/B7 comparisons and prespecified ablations | report macro F1, false-direct, NLL, Brier, coverage, singleton precision |
| Sep 11-13 | fresh closed-loop rollouts and failure taxonomy | OPEN, recognition, reroute, grasp, place, task success scored separately |
| Sep 14-15 | internal “submission-quality” freeze | one-command reproduction, table generated from immutable JSON, claim/no-claim memo |
| Sep 16-Oct 4 | robustness cycle | new seeds and prompt paraphrases without changing the frozen test result |
| Oct 5-Nov 6 | reviewer-mode replication | clean checkout reproduction and independent failure review |
| Nov 7-Jan 31 | high-risk method iteration | only changes justified by the frozen failure taxonomy |
| Feb 1-Mar 6 | camera-ready-style reproducibility freeze | code/data cards, hashes, environment lock, final negative results retained |

## Experiment matrix for the fixed drawer scenario

| ID | question | data/evidence | primary metric |
|---|---|---|---|
| E0 | Can frozen pi0.5 finish directly from the closed drawer? | live B0 rollouts | task success, wrong-object contact |
| E1 | Does OPEN expose the target? | live OPEN branch and evaluator replay | drawer opening, target pixels |
| E2 | Can pi0.5 use acquired information? | actual OPEN state followed by DIRECT | target grasp and final task success |
| E3 | Does the router use the prompt on identical RGB? | butter/cream-cheese prompt pairs per seed | paired route accuracy, prompt-swap accuracy |
| E4 | Does effect supervision help? | same-state executed candidate forks | B7 minus B6 route F1 and counterfactual ranking |
| E5 | Is the decision calibrated? | isolated calibration seeds | coverage, set size, singleton precision, false-direct |
| E6 | Which input is necessary? | no-prompt, last-frame, no-effect, no-calibration | delta in E3-E5 metrics |
| E7 | Does the full loop help? | B0 versus learned OPEN/reobserve/DIRECT | final task success per physical interaction |

## Current evidence and immediate decision

As of 2026-08-22, the software contracts pass, and a real OPEN action works.
Fresh executor qualification shows that DIRECT fails from both idealized and
actually opened states and frequently approaches or grasps the visible cream
cheese instead of the butter. Consequently:

- continue route/calibration experiments, because their hypotheses remain
  independently testable;
- treat full task success as executor-blocked until a frozen, public-history-only
  bridge or supported capability passes a preregistered qualification set;
- do not count fixture replay, ideal state transitions, or evaluator labels as
  learned method performance;
- do not spend the September cycle scaling the router if B6/B7 cannot beat the
  paired prompt counterfactual baselines.

## Daily rhythm

1. Morning: run one frozen gate before modifying code.
2. Midday: make one causal change tied to one failed metric.
3. Evening: rerun the paired comparison, save immutable JSON, and write a
   three-line keep/revert/next decision.
4. Every third day: reproduce from a clean process and audit GPU shutdown,
   hashes, split membership, and privileged-input count.

## 2026-08-22 paper-cycle outcome

The ten-seed qualification and executed-effect development cycle is complete.
OPEN mechanically succeeds 9/10 and acquires butter evidence 8/10, but
post-OPEN DIRECT produces 0/10 butter picks and 0/10 task successes. The
visible-object control passes pick (10/10) and fails terminal placement (3/10).
Executed effect supervision ties route-only in all five grouped folds, so the
effect head is rejected as a route contribution for this scenario.

The internal deadline deliverable is therefore the falsification-oriented
technical-report draft in [`paper/main.md`](../paper/main.md), not a broad
ICRA/RSS method claim. Any later method resurrection requires new primitives,
variable target/location compositions, positive empty/rejected effects, and a
fresh group-disjoint calibration/test set; the current negative result remains
frozen.
