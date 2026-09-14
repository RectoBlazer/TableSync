# TableSync demonstration pipeline

This is the first training-data milestone for the hackathon. It records the scripted
two-arm table relay as observations and actions suitable for imitation learning.
The pilot is a pipeline check, not enough evidence of a useful learned policy.

Completed pilot: **3/3 successful episodes, 9,000 frames** (6,000 training and 3,000
validation). All archives passed independent action replay. Maximum final object
position difference from the recorded run was **0.065 mm**. Training-reader checks
passed for chunk padding, split separation and training-only normalization.

## Record

```powershell
.\.venv\Scripts\python.exe scripts\collect_demonstrations.py
```

Default output: `data/relay_pilot/`. The command refuses to overwrite an existing
dataset. To create another run, pass `--output data/relay_next`. Defaults are two
training episodes (seeds 1000-1001) and one validation episode (seed 10000), separate
from development/evaluation seeds 0-19. Use `--train-episodes` and `--val-episodes`
to change counts. More episodes from this narrow distribution do not add new skills.

At each 20 ms control boundary, the recorder captures the current RGB image and
12 joint positions, then applies 12 absolute joint targets for the next 20 ms.
MuJoCo advances ten 2 ms physics steps during that interval. Both camera and action
sampling run at 50 Hz in simulation time; collection need not run in real time.
Earlier demos default to 500 Hz interpolated targets; recording deliberately uses
50 Hz held targets and repeats all physical success checks under that timing.

The camera is fixed and oblique, viewing both arms and the table. Its dataset key is
`observation_images_overhead`. Frames are RGB uint8, 224 x 224. Shadows and reflections
are disabled for faster rendering; collision geometry and physics are unchanged.

## Contents

Each successful episode contains `episode.npz`, `metadata.json`, `expert.log`, a
preview PNG and success metrics. Failed attempts retain metadata/logs but no training
archive. The root manifest records source hashes, package versions, joint order,
camera settings and train/validation membership. Interrupted runs are incomplete;
only successful episodes present in the manifest are loadable by the reader.

| Array | Shape | Meaning |
| --- | --- | --- |
| observation_state | 3000 x 12 | Measured joint angles, radians |
| observation_images_overhead | 3000 x 224 x 224 x 3 | RGB before action |
| action | 3000 x 12 | Absolute actuator targets, radians |
| timestamp | 3000 | Simulation seconds, starting at zero |
| stage, active_arm | 3000 each | Expert annotations, excluded from policy inputs |
| initial_qpos, initial_qvel, final_qpos | Full simulator vectors | Replay/debug only, excluded from policy inputs |

Joint order is all six Arm A joints then all six Arm B joints, each ordered as
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
The expert uses simulator object/contact information to generate demonstrations.
The training reader supplies only RGB and robot joint state as observations.

## Validate and read

```powershell
.\.venv\Scripts\python.exe scripts\validate_demonstrations.py
```

Validation checks hashes, shapes, timestamps, action limits and nonblank images,
then replays every stored action without invoking the expert's planning logic.
It checks trajectory agreement, collisions, final placement and table release.
The result is stored in `validation.json`.

`scripts/demonstration_dataset.py` supplies NumPy samples with normalized state,
RGB scaled to [0, 1], 100-step action chunks and padding masks. Chunks stop at the
episode boundary; padding repeats the final action. Normalization statistics use
only training episodes and are shared with validation. Do not use the validation
episodes to fit normalization or optimize model weights.

## LeRobot boundary

The compressed NumPy archives are our raw capture format, **not LeRobotDataset v3**.
LeRobot integration must use its dataset writer to create the supported metadata,
tabular and image/video layout; simply renaming these archives will not work.
The planned feature mapping is `observation_state` to `observation.state`,
`observation_images_overhead` to `observation.images.overhead`, and `action` unchanged.
See the [official LeRobot dataset documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3).

The official LeRobot export and ACT training scripts are now implemented; see
[the training guide](TRAINING.md). The raw archives remain the original capture format.

Still outstanding: closed-loop learned-policy evaluation,
OpenVINO deployment/benchmarking, airborne hand-off, camera/language task reasoning,
and the complete multi-step dinner-table scenario. Core Ultra demo hardware access
also remains unresolved. The relay dataset alone does not fulfill the challenge.
