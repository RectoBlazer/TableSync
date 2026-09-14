# TableSync

Current milestone: a **training trajectory reference + ACT residual controller passed 10/10 reserved two-arm table-relay trials**, with no monitored collision stops. The original direct-action ACT policy still needs improvement. See [results, limitations and viewer commands](docs/GRASP_RESULTS.md).

The reference trajectory alone also passed 10/10: these results do not yet show a
task-success benefit from ACT's learned corrections.

## Hackathon training-data pipeline

The first implementation step beyond the exercises is synchronized demonstration
collection. See [the dataset guide](docs/DATASET.md) for format, commands and limits.

```powershell
.\.venv\Scripts\python.exe scripts\collect_demonstrations.py
.\.venv\Scripts\python.exe scripts\validate_demonstrations.py
```

The collector records a small pilot at `data/relay_pilot/` and refuses to overwrite
it. If the pilot already exists, run just the validator. A different output path
creates another run. The pilot has also been exported to `data/lerobot_relay_pilot/`.
See [the ACT training guide](docs/TRAINING.md) for the isolated training environment,
export verification, GPU training command and checkpoint files.

## First exercise

In PowerShell:

```powershell
Set-Location 'C:\Users\ASUS\OneDrive\Bureau\AI-Infra'
.\.venv\Scripts\python.exe scripts\check_setup.py
.\.venv\Scripts\python.exe scripts\check_setup.py --view
```

The first command tests a falling cube and a tiny OpenVINO calculation on available CPU/GPU devices. This is an installation check, not a robotics benchmark.

The second command opens the scene. Close the viewer window when finished. Run it again to restart the falling cube.

## What each part means

- `.venv`: isolated Python packages for this project.
- `scenes/first_scene.xml`: MuJoCo scene description. The cube begins 0.4 meters above the floor; its half-size is 0.04 meters.
- `scripts/check_setup.py`: advances physics in 0.002-second steps and checks the inference runtime.
- OpenVINO: executes model calculations on Intel hardware. It does not simulate physics or train the policy in this exercise.

## Hardware boundary

Local development machine: Intel Core i9-12900H, Iris Xe graphics, NVIDIA RTX 3070 Ti Laptop GPU with 8 GB VRAM. This is not the Core Ultra Series 2/3 machine specified for the final demo in the supplied challenge brief. Final demonstration hardware access remains unresolved.

## Second exercise

```powershell
.\.venv\Scripts\python.exe scripts\dual_arm_demo.py
```

Or use the VS Code Run and Debug configuration **TableSync: two SO-101 arms**.
The program builds `scenes/dual_so101.xml` from the licensed upstream model and opens
the viewer. Arm A turns its base and opens its gripper, returns, then arm B repeats.
The cycle lasts ten simulated seconds. Close the viewer to stop.

To check without a window:

```powershell
.\.venv\Scripts\python.exe scripts\dual_arm_demo.py --check
```

The check covers finite joint positions, tracking error and arm-to-arm contacts for
this one trajectory. It does not validate grasping, shared reach or general collision
avoidance. See `vendor/README.md` for the pinned asset version and license.

### Concepts to learn

- A **joint** permits relative movement between robot links. Here it rotates.
- An **actuator** applies forces to drive a joint toward its target angle.
- `data.ctrl` contains the target angles for these position actuators.
- `data.qpos` contains the actual joint angles resulting from physics.
- Angles are in **radians**: 0.25 rad is approximately 14 degrees.
- Each arm has five positioning joints plus one gripper joint, for 12 controls total.

Read the `targets(t)` function first. It changes only the shoulder-pan and gripper
targets while the other targets remain at `HOME`. This is programmed motion, not AI.
The base layout is a lesson setup; shared reach and transfer geometry come next.

## Next milestone

Extend the validated grasp to placement targets and both arms before attempting a
handoff. Training follows successful scripted demonstrations.

## Third exercise

```powershell
.\.venv\Scripts\python.exe scripts\reach_demo.py
```

VS Code configuration: **TableSync: shared target reach**.
Arm B now faces inward. Green marks a target at `[0, 0, 0.20]` meters in table/world
coordinates; blue and orange mark A and B's gripper reference points. The green marker
is a visual site, not a physical ball. A reference marker may be hidden inside the green
marker when it reaches the target.

Each arm approaches for four seconds, holds for two, returns for four, then rests for
two. The full cycle takes 24 simulated seconds. Console messages identify each phase.

`inverse_kinematics()` finds joint angles from a desired 3D position using MuJoCo's
site Jacobian. It works on scratch simulation data; live motion still uses actuators.
`schedule()` sends smoothly changing joint targets, leaving the other arm at home.
This is a fixed turn-taking schedule, not a general dynamic workspace planner.

```powershell
.\.venv\Scripts\python.exe scripts\reach_demo.py --check --snapshot
```

Checks: each held reference point is within 5 mm, no inter-arm or arm-table penetration
on the tested trajectory, and finite joint positions. The initial run achieved less
than 0.6 mm error for both arms. Metrics and a rendered image are saved in `artifacts/`.
Unreachable targets cause an error rather than being silently accepted.

This lesson solves **position only**. It does not control the gripper orientation,
prove grasp reachability, evaluate all self-collisions, or implement object transfer.
The target should remain unchanged until grasp geometry and path validation are added.

## Fourth exercise

```powershell
.\.venv\Scripts\python.exe scripts\grasp_demo.py
```

VS Code configuration: **TableSync: physical grasp and lift**. The sequence runs once
in about 19 simulated seconds; the viewer remains open afterward. Restart to repeat.

Arm A starts above a 40 g blue block, approaches with open jaws, closes, confirms
contact on both jaws, lifts, holds for three seconds, lowers, opens and retreats.
Arm B remains parked. This tests the grasp building block, not the full bimanual task.

The object has a free joint: gravity, contact and friction determine its motion.
There are no equality welds or runtime object-pose assignments. Initialization places
the robot in a pregrasp configuration; subsequent motion is actuator-driven.

### Read the code

- `make_model()`: adds the block and contact settings to the two-arm scene.
- `pose_ik()`: seeks a position while also keeping the selected gripper orientation.
- `plan()`: defines approach, close, lift, hold, lower, release and retreat targets.
- `grip_contacts()`: checks actual normal contact forces against both jaws.
- `run()`: sends targets through physics and checks outcomes.

The initial pickup pose and block orientation are adapted from the upstream model's
`scene_box.xml` example. The chosen 4 cm gripper lift respects wrist limits; a 12 cm
vertical lift with the same orientation did not solve. Object center rise is about
3.7 cm after grasp settling. The block dimensions are 4 by 4 by 6 cm, with its initial
orientation putting a 4 cm dimension vertically.

### Contact model settings and limitations

Gripper torque is limited to +/-0.3 Nm, with position gain 20 and velocity gain 1.
Jaw geometry and original friction are retained. This scene uses 50 main solver
iterations and 5 NoSlip iterations to suppress soft-contact drift. NoSlip is MuJoCo
contact post-processing, not an object attachment; results still depend on the
chosen simulation settings and are not hardware validation.
Reference: https://mujoco.readthedocs.io/en/latest/modeling.html

```powershell
.\.venv\Scripts\python.exe scripts\grasp_demo.py --check --snapshot
```

The check requires confirmed two-jaw contact before lift, block center above 5 cm
throughout the three-second hold, two-jaw contact during more than 90% of hold samples,
less than 2 mm hold-height drift, no inter-arm or arm-table penetration, and release
onto the table. The initial passing run had 100% two-jaw hold contact and less than
0.01 mm height drift. A negative control with the gripper left open was rejected
before lift. Metrics and the lifted snapshot are stored in `artifacts/`.

This is one known object pose and a scripted sequence using simulator contact
information. It does not establish randomized grasp success, camera-based grasp
selection, learned control, or a bimanual transfer.

## Fifth exercise

```powershell
.\.venv\Scripts\python.exe scripts\place_demo.py
```

VS Code configuration: **TableSync: pick and place**. The sequence takes about
23 simulated seconds and runs once. Arm A grasps the block, raises it, carries it
4 cm toward the table center, holds, lowers until table support is confirmed,
opens and retreats. The green square is a nonphysical destination marker.

`OFFSET` defines the desired displacement; `plan()` inserts the carry and placement
poses into the existing grasp skill. `require_table_support()` checks contact force
before opening. `assess_placement()` checks the object's final center, not just the
robot's commanded position. Leave the offset unchanged until the new path is tested.

```powershell
.\.venv\Scripts\python.exe scripts\place_demo.py --check --snapshot
```

The initial run finished 10.3 mm from the destination center (15 mm tolerance),
42.5 mm from its starting center, with confirmed table contact before release.
The placement check rejects a block left at its starting position. The established
grasp checks still apply, and results are written to `artifacts/place_check.json`.

This remains a scripted skill for a known initial pose, with Arm B parked. The next
integration step is to give Arm B a validated manipulation skill and sequence both
arms toward a multi-step task. These lessons do not yet constitute the hackathon MVP.

## Sixth exercise

```powershell
.\.venv\Scripts\python.exe scripts\bimanual_demo.py
```

VS Code configuration: **TableSync: both arms pick and place**. The sequence runs once
in about 52 simulated seconds; the viewer remains open afterward.

Arm A picks up the blue block, places it at its green marker and parks. Arm B then
does the same with the orange block. The second arm uses the same local joint skill
with its base and object rotated 180 degrees. Both objects move through simulated
contact and friction.

The controller confirms two-jaw contact before lifting and table support before
release. It checks each placement and checks both again after Arm B finishes.
Only one arm moves at a time: this is a fixed conservative sequence, not a general
shared-workspace planner or a hand-off.

```powershell
.\.venv\Scripts\python.exe scripts\bimanual_demo.py --check --snapshot
```

The initial run placed both blocks within 10.4 mm of their targets (15 mm tolerance),
with two-jaw contact throughout carry/hold and no detected inter-arm or arm-table
penetration. These checks cover the tested trajectory, not all possible collisions.
Metrics are saved to `artifacts/bimanual_check.json`; images show each held block
and the final scene. This remains scripted control with known object poses.
Shared-object transfer, language/camera reasoning, learned control, OpenVINO policy
deployment and randomized evaluation remain to be implemented.

## Seventh exercise

```powershell
.\.venv\Scripts\python.exe scripts\relay_demo.py
```

VS Code configuration: **TableSync: shared object table relay**. Runs once in about
60 simulated seconds. A carries the blue block to the green transfer marker and
parks. B then grasps that same block and returns it to the orange-marked start area.
This scene places B closer to the transfer station; earlier exercise scenes retain
their original layout. There is one free object, with no runtime pose assignments
or welds.

B can begin only after the block has table support, A has released it, and A's joints
are within 0.08 radians of the parked pose. B corrects its pickup targets using the
measured object XY position and inverse kinematics. This uses simulator ground truth,
not camera perception. The uncorrected nominal pickup failed the grasp gate because
A's placement settled roughly 1 cm off-center.

```powershell
.\.venv\Scripts\python.exe scripts\relay_demo.py --check --snapshot
```

The passing run achieved 10.3 mm transfer placement error and 1.7 mm final placement
error, with two-jaw contact throughout both carry/hold phases. No inter-arm or
arm-table penetration was detected on this trajectory. Metrics are saved in
`artifacts/relay_check.json`, with snapshots of each arm holding the same object.

This is a table-supported relay, **not an airborne hand-off**. It demonstrates
dependent task sequencing, measured pickup correction and exclusive access to the
transfer area. General collision avoidance, direct hand-off, camera/language input,
learned policies and randomized robustness remain outstanding.

## Eighth exercise: measure reliability

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_relay.py
```

VS Code configuration: **TableSync: evaluate 10 relay seeds**. This runs without a
viewer, prints one PASS/FAIL line per trial and saves a report after every trial.
Each seed reproducibly selects an independent starting XY offset within +/-3 mm,
a block mass between 30 and 50 g, and block sliding friction between 0.8 and 1.2.
The model compiler recalculates object inertia for the selected mass. Jaw friction,
object orientation, lighting, robot layout and task remain fixed.

A seed is an integer that lets you recreate the same random choices. For example,
to inspect seed 3 visually:

```powershell
.\.venv\Scripts\python.exe scripts\relay_demo.py --seed 3
```

To evaluate a separate batch:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_relay.py --start-seed 10 --seeds 10
```

Read `artifacts/relay_eval_0_10/summary.json` for the first batch's success rate,
parameters and per-arm metrics. Each seed folder contains a stage log and result;
failures retain their reason, elapsed simulation time and last object position.
All requested trials are attempted even when a control check fails. The process
returns exit code 1 if any trial fails. Rerunning the same batch replaces its report.

Success requires both arms to grasp and carry the block, meet the 15 mm placement
tolerance and release onto the table, with no detected inter-arm or arm-table
penetration. This is a narrow robustness test of the scripted table relay using
simulator feedback. It does not establish airborne hand-off success, visual
generalization, a learned policy, or inference performance. Execution time here
includes simulation and is not an OpenVINO benchmark.

Initial seeds 0-9: **9/10 successful (90%)**. Seed 7 failed A's two-jaw grasp check
at 5 simulated seconds; the lift was blocked. Successful final placement errors
ranged from approximately 0.7 to 4.1 mm. This small batch identifies a grasp
sensitivity to investigate; it is not a statistical guarantee of 90% reliability.

## Ninth exercise: correct the donor pickup

```powershell
.\.venv\Scripts\python.exe scripts\relay_demo.py --seed 7
```

Arm A now measures the initial block XY offset, adjusts its skill poses through
inverse kinematics, and moves to the corrected pregrasp before descending. Arm B
still measures the block again after A places it. Each arm starts from the unchanged
nominal plan, so A's correction cannot accidentally accumulate into B's correction.
The terminal and JSON metrics expose the correction for each arm in the IK frame.
B's XY correction has the opposite sign because its base is rotated 180 degrees.

The former failing seed 7 now passes, with A's measured correction of about
0.75 mm in X and 2.38 mm in Y. This is simulator-state feedback, not learned or
camera-based perception. Contact checks and placement tolerances are unchanged.
The corrected motion adds three seconds of pregrasp alignment for A.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_relay.py --seeds 20
```

This checks the original seeds 0-9 plus new seeds 10-19. Results are stored in
`artifacts/relay_eval_0_20/summary.json`. The original baseline batch remains in
`artifacts/relay_eval_0_10/` until that batch is rerun; its summary was also saved as
`artifacts/relay_baseline_0_10.json` for comparison.

Corrected controller results: **20/20 passed**, including all 10 original seeds and
10 new seeds. Maximum final placement error was approximately 5.1 mm against the
unchanged 15 mm tolerance. These results cover only the stated modest position,
mass and friction variations; they do not establish general reliability.
