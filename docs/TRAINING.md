# Local LeRobot / ACT pilot

The simulation environment remains `.venv`; training uses `.venv-train` with
Python 3.12, LeRobot 0.4.4, PyTorch 2.10.0+cu128 and torchvision 0.25.0+cu128.
`requirements-training-lock.txt` records the complete installed dependency set.
CUDA execution was verified on the RTX 3070 Ti Laptop GPU. No Hub upload is used.

Completed pilot: all 9,000 frames exported and reloaded, with raw-data boundary
comparisons and chunk-padding checks passing. The 15.6M-parameter ACT model completed
300 steps in about 90 seconds. On 30 sampled frames from the held-out episode,
action-chunk MAE fell from **0.4804 to 0.0696 radians**. Checkpoint reload produced
identical predictions. These are offline prediction results, not robot task success.

## Recreate the training environment

```powershell
C:\Python312\python.exe -m venv .venv-train
.\.venv-train\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv-train\Scripts\python.exe -m pip install lerobot==0.4.4
.\.venv-train\Scripts\python.exe -m pip install mujoco==3.13.0
.\.venv-train\Scripts\python.exe -m pip install -r requirements-training-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
.\.venv-train\Scripts\python.exe -m pip check
```

These are the [official PyTorch version combinations](https://pytorch.org/get-started/previous-versions/).
The scripts configure project-local Hugging Face and Torch caches before imports.
Hub access is disabled. ResNet18's official ImageNet weights must be present in
`artifacts/torch-cache/hub/checkpoints/`; the first model initialization downloads
them from PyTorch if absent and network access is available.

## Export

```powershell
.\.venv-train\Scripts\python.exe scripts\export_lerobot.py
```

Output: `data/lerobot_relay_pilot/train` and `val`, each a separate LeRobotDataset v3.
Use `--resume` only to reuse a complete split after an interrupted export; incomplete
splits require a fresh `--output` path. Images are stored through LeRobot's image
feature, avoiding video decoding dependencies for this small pilot. Raw archives
are retained. Export checks compare episode boundary images, states and actions
to the raw files and verify that action chunks pad at episode boundaries.

This uses the [official LeRobot dataset writer](https://huggingface.co/docs/lerobot/lerobot-dataset-v3),
including `save_episode()` and `finalize()`. `export_report.json` records counts;
`integrity_report.json` records the boundary checks.

## Train

```powershell
.\.venv-train\Scripts\python.exe scripts\train_act.py
```

Default: 300 optimizer steps, batch size 4, random seed 42, and the official LeRobot
ACT policy with a pretrained ResNet18 vision backbone. The smaller transformer uses
256 hidden dimensions, two encoder layers, one decoder layer and two VAE encoder
layers. It predicts 50 future actions (one second at 50 Hz). Configuration stores
10 actions per planned inference call, subject to subsequent closed-loop validation.

The policy sees RGB and 12 measured joint angles; it predicts 12 absolute joint
targets. It has no simulator object poses, contact states, stage labels or language
encoder. Only the training split supplies normalization statistics and gradients.
Validation samples 30 fixed frames across the held-out episode and measures masked
action-chunk mean absolute error, both normalized and in radians, before and after
training. This is offline imitation error, not task completion or collision safety.

Outputs in `outputs/act_pilot/`:

- `pretrained_model/`: model, configuration and preprocessing/postprocessing assets.
- `training_report.json`: configuration summary, loss samples, validation metrics,
  timing, GPU memory and checkpoint reload comparison.
- `training_state.pt`: optimizer and Torch RNG state for a future resume implementation.
- `progress.json`: partial training log.

Output directories are never overwritten. For a fresh run use
`--output outputs/act_next --steps 1000`. There is no resume CLI yet. A saved checkpoint
is reloaded and checked against the in-memory policy, with a finite 50 x 12 action
chunk required from the restored processors.

Improvement experiments can use `--init-checkpoint PATH` to warm-start weights with
a fresh optimizer (this is not an exact training resume). `--grasp-weight 4` samples
approach/closure frames four times as often as other frames, using the raw training
annotations only for sampling. Episode order is checked against the LeRobot export;
these annotations are never supplied to the policy. See [the experiment log](GRASP_EXPERIMENTS.md).

Next integration: run the learned policy in MuJoCo with fresh camera observations,
measure completion on reserved seeds, and then export/benchmark OpenVINO inference.
Closed-loop evaluation is now implemented: [results and commands](CLOSED_LOOP.md).
The pilot failed the tested relay trials despite its improved offline error.
The three-episode pilot and offline losses do not establish a deployable controller.
Airborne hand-off and the full dinner-table task remain separate work.

## Current hybrid experiment

The numeric action cache in `fast_lerobot_dataset.py` preserves LeRobot image
decoding, episode boundaries and padding while avoiding repeated conversion of
50 tiny action rows per sample.

The corrective dataset contains six training and two validation episodes. To
reproduce the residual training run from its saved absolute-action checkpoint:

```powershell
.\.venv-train\Scripts\python.exe scripts\train_act.py --dataset data/lerobot_relay_corrective --raw-dataset data/relay_corrective --init-checkpoint outputs/act_balanced/pretrained_model --task-time --trajectory-prior --steps 1000 --batch-size 32 --lr 0.0001 --save-every 250 --output outputs/act_residual_reproduction
```

This uses a fresh optimizer. `--task-time` adds elapsed episode time / 60 to the
observations. `--trajectory-prior` computes a mean action trajectory from training
episodes only, then trains ACT on residual targets. The state projections gain an
initially zero clock column, and the action head starts at zero residual. Residual
standard deviations have a 0.02 rad floor. A checkpoint's reference and processors
must travel with its weights; the evaluator adds the predicted residual to that
reference at runtime. This is a different controller from direct-action ACT.

Checkpoint 250 was selected using physical development results before the final
evaluation, even though training continued to 1000. Do not substitute another
checkpoint and attribute the selected checkpoint's success rate to it. See
[the current grasp results](GRASP_RESULTS.md).
