# ACT closed-loop evaluation

The later hybrid controller has now completed 10/10 final trials. See
[the current results and viewer commands](GRASP_RESULTS.md). The pilot results
below remain as the baseline; they are not overwritten by the hybrid result.

The pilot checkpoint was tested with fresh rendered images and measured robot joint
angles. Its actions alone drive both arms after the fixed pregrasp initialization.
The scripted expert provides no runtime targets, stage transitions or recovery.
Simulator contact and object state are used only for scoring and collision stops.

## Run

```powershell
.\.venv-train\Scripts\python.exe scripts\evaluate_act.py --seeds 0 1 2 --output artifacts/act_eval_next
```

To watch one run:

```powershell
.\.venv-train\Scripts\python.exe scripts\evaluate_act.py --view --seeds 0 --output artifacts/act_view_01
```

Choose a new output directory each time; existing results are not overwritten.
The viewer closes when the task succeeds, hits a stop condition or reaches the
60-second simulated time limit. A failed task produces exit code 1 and saved reports.

ACT predicts 50 actions from a fresh 224 x 224 RGB observation and 12 joint angles.
The evaluator executes the first 10 actions at 50 Hz, then queries again (5 Hz in
simulation time). Actions are bounded by actuator limits; clipping is counted.
There is no scripted action fallback. Camera parameters and rendering flags match
the demonstrations. Collision checks run at every 2 ms physics step.

Success requires four ordered events: A holds the block with both jaws above 5 cm
for 0.2 seconds; A releases it on the table within 15 mm of the transfer marker for
0.2 seconds; B holds it above 5 cm for 0.2 seconds; B releases it within 15 mm of the
final marker for 0.5 seconds. A final-position-only test would incorrectly reward a
stationary block, since this relay ends near its starting point.

## Pilot results

- Reserved seeds 0, 1, 2: **0/3 successful**, no first sustained grasp/lift.
- Diagnostic training seed 1000: **0/1 successful**, also no sustained grasp/lift.
- Block-center peak heights on reserved seeds: 32.0, 32.8 and 29.0 mm, below 50 mm.
- No actuator values were clipped in these runs; all ended at the time limit.
- The matched scripted expert passed all four monitor events on seed 0 at 50 Hz in
  the same training environment. The monitor also rejects stationary and wrong-order
  synthetic traces in `tests/test_act_monitor.py`.

Reports: `artifacts/act_closed_loop_pilot/summary.json` and
`artifacts/act_training_seed_diagnostic/summary.json`. Each seed folder contains the
joint/action trajectory, object trajectory, result JSON and last camera observation.
The matched expert's physical metrics are in `artifacts/act_matched_expert/`.

The pilot's lower offline error did not translate into reliable grasping. Failure
on a training seed shows that this is not just generalization to new randomization.
Further training and data work must be evaluated with this runner; averaging joint
prediction error alone is insufficient. The exact relative contributions of limited
training, data coverage and accumulated control error have not yet been isolated.

Inference timing in the JSON includes preprocessing, network execution and action
postprocessing, but excludes rendering and physics. It is diagnostic NVIDIA timing,
not an OpenVINO or Core Ultra benchmark.
