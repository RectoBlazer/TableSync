# Learned grasp improvement log

Target for this work: stable physical grasp/lift by both arms and at least 8/10
complete table relays on a final set of narrowly randomized, unseen seeds, with
no inter-arm or arm-table collision stops. This is a development acceptance target,
not evidence about airborne hand-off or broad generalization.

Development checks use training seed 1000 and seeds 0-2. Final evaluation reserves
seeds 20000-20009. Do not use that final set for choosing changes. If it fails and
development continues, disclose the exposure and reserve a new final set.

Keep scoring, camera, task, tolerances and 50 Hz action/5 Hz observation timing
unchanged for the initial training experiment.

| Experiment | Change | Offline evidence | Physical result |
| --- | --- | --- | --- |
| act_pilot | 300 steps, two training episodes | Held-out chunk MAE 0.0696 rad | 0/3 on seeds 0-2; 0/1 on training seed 1000 |
| act_3000 | 3,000 steps, same architecture/data/settings | Held-out chunk MAE 0.0243 rad | 0/4 on development seeds 1000, 0, 1, 2; no sustained A lift |
| act_grasp_weighted | Warm-start act_3000; 1,500 steps with 4x approach/closure sampling and fresh optimizer | Held-out chunk MAE 0.0214 rad | 0/4 on seeds 1000, 0, 1, 2; tends to remain above the block |

Step three is collecting four additional training episodes (2000-2003) and one
validation episode (11000), with each non-gripper joint initialized within +/-0.04
radians of its nominal pose. The expert commands the normal task, correcting those
offsets through physics. Original episodes are retained in the combined dataset.

Corrective dataset completed: six training and two validation episodes, 24,000
frames. All passed action replay and LeRobot export integrity checks. A live initial
render for seed 1000 also matched the original training RGB exactly.

The corrective fine-tune starts from act_grasp_weighted for 2,000 steps at learning
rate 3e-5. Recovery/pregrasp, approach and closure frames receive sampling weight 3.
Nearly stationary targets with small tracking error receive multiplier 0.2; frames
with initial corrective tracking error remain eligible at full weight. Intermediate
checkpoints are saved every 500 steps for physical development tests.

Run order: longer training first; if unsuccessful, inspect action errors and give
approach/closure more training weight; then add corrective demonstrations if needed.
No scripted action fallback is permitted in the learned-policy success score.

## Follow-up experiments

- Corrective checkpoint 500 genuinely achieved sustained A grasp/lift on seeds
  1000 and 0, then stopped on arm-table contact. Full relay: 0/2. Replaying seed
  1000 identified A's gripper contacting the table at 22.236 s (approximately
  4 micrometres penetration); the collision criterion was not relaxed.
- Corrective checkpoint 1000 stalled: 0/4 on development seeds. Training was
  interrupted after that saved checkpoint to repair slow numeric action loading.
- A cache of numeric actions preserved sampled image/state/action/padding values
  across episode boundaries. A further 1000 steps from checkpoint 1000 completed
  in 55.8 seconds, with held-out MAE 0.0142 rad but 0/4 full relays. Executing
  all 50 chunk actions instead of 10 also failed 0/4.
- Balanced fine-tuning from the physically better corrective checkpoint 500 used
  batch 32, learning rate 1e-5 and 1000 updates. MAE reached 0.0110 rad; intermediate
  checkpoints at 250/500/750/1000 still failed the two development trials each.

The next experiment explicitly adds elapsed episode time divided by 60 as a
thirteenth state input. This addresses ambiguity between long holds and subsequent
motions. It is a **different, clock-conditioned policy**, not evidence that the
RGB-and-joints baseline succeeded. The clock is observable during deployment;
expert phase labels, object coordinates and target actions are still excluded
from inference. Timing dependence also limits resilience to delayed task progress.
The input projections are expanded with an initially zero clock column, retaining
the previous model's initial predictions. It trains on the corrective training
split, with fixed clock scaling and the same untouched validation split.

Clock-only addition at 1000 and 2000 updates still failed both development trials
(1000 and 0). An independent diagnostic replayed the pointwise mean of the six
**training** action trajectories: 4/4 complete relays on development seeds
1000, 0, 1 and 2, with no monitored collision stops. This is open-loop trajectory
replay, not a successful vision policy. Its report is
`artifacts/mean_trajectory_development/summary.json`.

The subsequent hybrid experiment uses that same training-derived trajectory as
a reference and trains ACT to predict residual joint targets from RGB, current
joints and elapsed task time. It does not call the scripted expert at runtime.
It is **not** the original end-to-end absolute-action ACT policy: much of the
motion comes from the reference. The reference-only baseline must be reported
alongside it, and no benefit from learned corrections may be claimed without an
observed improvement over that baseline. Residual normalization and the reference
use training episodes only. A numerical check reconstructed original action chunks
at both episode boundaries with maximum discrepancy below 5e-10 rad.

Residual checkpoint 250 passed all four development trials. It was frozen and
hashed in `artifacts/grasp_release_candidate.json` before testing seeds 20000–20009.
**Hybrid final result: 10/10**, with both sustained lifts, both placements and no
monitored collision stops. All frozen checkpoint hashes remained unchanged.
The exact same final seeds also gave **10/10 for the reference alone**. Therefore
the experiment establishes reliable narrowly randomized table relays for the
hybrid, but does not establish an advantage from its learned corrections or solve
the original standalone ACT objective. Full reports and viewer commands are in
[GRASP_RESULTS.md](GRASP_RESULTS.md).

The residual training continued to step 1000, with held-out residual MAE decreasing
from 0.001532 to 0.001435 rad. That later checkpoint was **not** selected or tested
on the final seeds; its offline metric must not be attributed to checkpoint 250.

The final 3000-step clock-conditioned direct-action checkpoint was also tested:
0/2 on development seeds 1000 and 0, with no sustained A lift despite validation
MAE 0.00815 rad. This closes the clock-only experiment; adding a clock alone did
not resolve the direct policy's stalling. The hybrid result must remain separate.

Final checks: all ten hybrid trial seed IDs match the predeclared list; all frozen
asset hashes match; both final controllers have every ordered physical milestone;
the three monitor regression tests pass; the viewer launcher parses successfully.
The compact final comparison is `artifacts/grasp_final_comparison.json`.

The stage diagnostic for act_3000 found A approach wrist-roll error about 0.056 rad,
and A closure gripper-target error about 0.151 rad (teacher-forced samples from seed
1000). These specific errors remain material despite the lower overall average.
