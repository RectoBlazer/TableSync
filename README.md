# TableSync — teammate handoff and submission plan

TableSync is our entry for the **AI Infra Summit Hackathon, Intel Physical AI Online Challenge**. The intended demo is a language-and-vision-driven dinner-table workflow in MuJoCo, using two simulated SO-101 robot arms and OpenVINO inference on Intel hardware.

**Current position:** we have a working two-arm manipulation baseline, training and evaluation tooling, and a hybrid controller that completed 10/10 reserved table-relay trials. We do **not** yet have the complete dinner-table task, multimodal planner, dashboard, OpenVINO policy deployment or submission video.

We are continuing development and benchmarking on the available **Intel Core i9-12900H and Iris Xe iGPU**. The NVIDIA RTX 3070 Ti Laptop GPU is used for training and the current PyTorch policy runs. Label the actual hardware accurately; no Core Ultra or NPU results have been produced.

## What has been built

The project progressed from basic simulation checks to physical grasping, placement, two-arm coordination, demonstration collection and learned-policy experiments.

| Component | Implemented and verified |
| --- | --- |
| Simulation | Two SO-101 arms in MuJoCo, with physical gripper contacts and a movable block. Objects are not attached to the grippers with artificial welds. |
| Scripted expert | Arm A picks up a block and places it in a shared transfer area; Arm B picks up the same block and returns it to the final location. |
| Coordination checks | Ordered grasp/lift/placement milestones, plus inter-arm and arm-table collision stops. A stationary block cannot pass the success test. |
| Demonstrations | Six training and two validation episodes: 24,000 frames of images, joint observations and action targets. Action replay and LeRobot export checks passed. |
| Policy training | LeRobot ACT training, saved checkpoints/processors, validation metrics and multiple grasp-improvement experiments. |
| Working controller | A mean trajectory computed from training demonstrations, plus small learned ACT corrections from RGB, joint angles and elapsed task time. |
| Evaluation | Per-seed results, physical milestone timestamps, joint/action traces, object trajectories and final camera images. |
| Intel setup | OpenVINO installation and basic inference checks on the available machine. Robotics-model deployment through OpenVINO is still pending. |

### What the results actually establish

| Controller | Result |
| --- | --- |
| Original direct-action ACT | Still unreliable; more training reduced prediction error without producing a reliable complete relay. |
| Selected hybrid: training reference + ACT corrections | **4/4 development trials, then 10/10 reserved trials**, with no monitored collision stops. |
| Training reference alone | **10/10 on the same reserved seeds**. No task-success benefit from ACT corrections has been established. |

The current success is a **table-supported block relay**, not an airborne hand-off or a complete dinner-table workflow. Randomization is modest: initial XY placement ±3 mm, mass 30–50 g and sliding friction 0.8–1.2. Object shape, lighting, background and camera remain fixed. The controller uses a task clock; recovery after a failed grasp or moved object is not established.

Actions execute at 50 Hz of simulation time, with a new prediction every 10 actions. This is not a measured OpenVINO real-time performance claim.

Detailed evidence: [grasp results](docs/GRASP_RESULTS.md), [experiment history](docs/GRASP_EXPERIMENTS.md), [dataset guide](docs/DATASET.md), [training guide](docs/TRAINING.md), and [closed-loop evaluator](docs/CLOSED_LOOP.md).

## Ownership for the two-person team

### Project lead — robotics, backend and Intel deployment

The project lead, working with Codex, owns:

1. A small backend interface around the simulator/controller: start, stop, status, camera frames and results.
2. Extending the relay into a coherent dinner-table workflow with multiple placements and an explicit transfer or complementary two-arm action.
3. Language-and-camera reasoning that interprets instructions, chooses actions and checks task progress.
4. OpenVINO conversion, inference integration and Intel CPU/iGPU benchmarking.
5. Final randomization, ten-seed evaluation, model packaging and technical documentation.

### Teammate — dashboard, demo recording and submission presentation

**Your first task is the local demo dashboard. You do not need to train models or install the robotics stack to start.**

Build a simple interface with:

- A natural-language command input and Start/Stop controls.
- A simulation camera feed.
- The interpreted plan, current step and Arm A / Arm B activity.
- Clear idle, running, succeeded, failed and stopped states, including collision-stop messages.
- A ten-seed results table that includes failures.
- A benchmark panel showing hardware, runtime/device, model precision and measured latency/throughput.

Start with **clearly labeled mock data**. Keep backend communication in one adapter so sample responses can be replaced without rewriting the UI. Do not add authentication, a database or cloud deployment. Do not display fabricated reasoning or benchmark numbers as live results.

After the dashboard is connected, own the recording and editing of the demo video, screenshots, presentation copy and preparation of the submission form. The project lead supplies the verified technical claims and results.

If helping with backend integration, own the dashboard adapter and presentation of incoming status. Keep simulator control, model inference and success scoring with the project lead to avoid overlapping edits.

## Interface we need to agree on first

**This contract is proposed; a backend API is not implemented yet.** Agree on the payloads before building transport-specific UI code.

| Operation | Minimum information |
| --- | --- |
| Start | Instruction and seed; backend returns a run ID or a clear error. |
| Stop | Run ID; backend acknowledges whether the run has stopped. |
| Status | Run ID, state, interpreted plan, current step, arm activity and errors. |
| Camera | Latest simulated camera frame associated with the run. |
| Result | Seed, success/failure, verified milestones and collision-stop information. |
| Benchmark | Actual hardware, model/component, device, precision, latency, throughput and measurement conditions. |

Mock runs should cover success, failure, stopping and a disconnected backend. Display the observed plan and status supplied by the backend; do not infer successful manipulation from animation timing.

## Remaining work before submission, in order

| Priority | Work | Owner | Completion condition |
| --- | --- | --- | --- |
| 1 | Confirm the submission cutoff, form fields and video constraints; agree on the interface above. | Both | One shared checklist and agreed sample payloads. |
| 2 | Implement the backend wrapper and dashboard in parallel. | Lead: backend; teammate: UI | Real start/stop, camera and status work through the dashboard. |
| 3 | Complete a small dinner-table workflow with meaningful two-arm coordination. | Lead | Multiple task steps, explicit transfer/complementary action, shared-workspace sequencing and verified final placement. |
| 4 | Integrate genuine language-and-vision reasoning. | Lead | The instruction and camera observations influence the plan; state checks govern progression. |
| 5 | Integrate OpenVINO and benchmark Intel CPU/iGPU. | Lead | Actual supported model components run through OpenVINO; task behavior and performance are measured. |
| 6 | Freeze the integrated system and evaluate ten fresh randomized seeds. | Lead | Reports cover the final task and inference runtime, including failures. Existing relay results cannot substitute for this test. |
| 7 | Record and edit the final demo. | Teammate, with lead operating/debugging | Command, scene variation, both arms, outcomes and actual benchmark results are visible. |
| 8 | Test reproducibility and complete submission materials. | Lead: technical package; teammate: presentation/form preparation | A fresh setup can reproduce the demo; every submitted link works. |

The dashboard can be built while robotics and inference work continue. Recording depends on the integrated system and verified results. If dashboard integration slips, record the simulator with a simple command/status panel rather than delay the whole submission.

For this deadline, defer further standalone-ACT training, fluid simulation, broad object libraries, elaborate UI polish and NPU/INT8 experiments unless the complete submission path already works. Quantization is useful only if supported and task quality is preserved.

## Submission package and video

The challenge brief lists five required deliverables:

- **Reproducible GitHub repository:** setup, dependencies, scene/assets, training, inference, evaluation and demo commands.
- **Reproducible MuJoCo simulation:** the final dual-arm scenario and its randomization/evaluation configuration.
- **Intel inference benchmark script:** device selection, precision, latency and throughput on the actual available hardware. Document any difference from the brief's stated target hardware.
- **Demonstration video:** successful execution evidence across ten randomized seeds, with commands, scene variations and outcomes easy to verify.
- **Technical README / architecture summary:** model choices, coordination, training, robustness, OpenVINO optimization and hardware mapping.

Also package or provide download instructions for required model weights, processors, trajectory references and robot assets. Include licenses and saved evaluation/benchmark reports. Check the submission portal for additional fields, access requirements and video limits; those have not been confirmed here.

Suggested video sequence: problem and command → initial scene → visible perception/plan → one complete two-arm workflow → labeled montage of all ten seeds → results and Intel/OpenVINO benchmark. Label accelerated footage and distinguish reference motion from learned corrections. Do not claim that the current hybrid outperforms its reference-only baseline.

## Repository map and teammate starting point

| Location | Purpose |
| --- | --- |
| `scripts/` | Simulation, expert control, data collection, training, evaluation and viewer launcher. |
| `scenes/` | MuJoCo scene files. |
| `tests/` | Regression checks, including ordered task-success scoring. |
| `docs/` | Dataset/training instructions and experiment evidence. |
| `vendor/README.md` | Robot asset provenance and version information. |
| `requirements-setup.txt` | Simulation/OpenVINO setup dependencies. |
| `requirements-training-lock.txt` | Training-environment dependency snapshot. |

**A Git clone does not include the local training environment, datasets, checkpoints, result artifacts or downloaded robot asset tree.** These are excluded by `.gitignore`. Links in the experiment documents to `data/`, `outputs/` and `artifacts/` describe files on the development machine, not files automatically delivered by GitHub.

For the teammate: create a `demo-dashboard` branch, start the UI in a dedicated `frontend/` directory, add its own setup instructions, and work against labeled sample responses. The `frontend/` directory is a planned addition, not an existing application. Use small commits and coordinate changes to shared interfaces before merging.

To watch the current hybrid on the development machine, where the environment, robot assets and checkpoint are present:

```powershell
.\scripts\watch_learned_relay.ps1
```

See [the results guide](docs/GRASP_RESULTS.md) for the direct Python command and required checkpoint files. A fresh clone needs those dependencies and assets before this launcher will work.

The existing [team proposal](TableSync_Team_Proposal.docx) records the earlier plan. This README and the experiment reports describe the current implementation and remaining work.
