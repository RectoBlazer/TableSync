"""Record synchronized RGB/state/action episodes from the checked relay expert.

Raw TableSync format, not a LeRobotDataset. No publishing or training is performed.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import mujoco
import numpy as np
from relay_demo import make_model, plan, execute, HOME, ROOT
from dual_arm_demo import JOINTS, write_png

FPS = 50
STAGES = ['pregrasp', 'approach', 'close', 'lift', 'carry', 'hold', 'lower', 'release', 'retreat', 'park']
TASK = 'Use arm A to move the blue block to the green transfer spot, then use arm B to return it to the orange start area.'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def collect(seed, split, root, initial_jitter=0.):
    out = root/split/f'seed_{seed:06d}'
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    metadata = {'seed': seed, 'split': split, 'success': False, 'fps': FPS,
                'task': TASK, 'stage_names': STAGES, 'format': 'tablesync-raw-v1'}
    model = make_model(seed); initial, steps = plan(model)
    data = mujoco.MjData(model); data.qpos[:6] = initial; data.qpos[6:12] = HOME
    data.ctrl[:] = data.qpos[:12]
    if initial_jitter:
        rng = np.random.default_rng(seed+100000)
        ids = np.r_[np.arange(5), np.arange(6,11)]
        jitter = rng.uniform(-initial_jitter,initial_jitter,len(ids))
        data.qpos[ids] += jitter
        metadata['initial_joint_jitter_rad'] = jitter.tolist()
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy(); initial_qvel = data.qvel.copy()
    metadata['randomization'] = {'initial_block_position_m': data.body('block').xpos.tolist(),
        'mass_kg': float(model.body_mass[model.body('block').id]),
        'friction': model.geom_friction[model.geom('block_geom').id].tolist()}
    camera = mujoco.MjvCamera(); camera.lookat[:] = [-.08, 0, .1]
    camera.distance = .95; camera.azimuth = 90; camera.elevation = -35
    metadata['camera'] = {'lookat': camera.lookat.tolist(), 'distance': camera.distance,
                          'azimuth': camera.azimuth, 'elevation': camera.elevation}
    images = []; states = []; actions = []; timestamps = []; stages = []; owners = []
    with (out/'expert.log').open('w', encoding='utf-8') as log, redirect_stdout(log):
        try:
            with mujoco.Renderer(model, height=224, width=224) as renderer:
                def record(model, data, arm, stage):
                    # Forward updates camera geometry to q(t). Targets are held for the next 20 ms.
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                    images.append(renderer.render().copy())
                    states.append(data.qpos[:12].astype(np.float32).copy())
                    actions.append(data.ctrl.astype(np.float32).copy())
                    timestamps.append(data.time); stages.append(STAGES.index(stage)); owners.append(arm == 'B')
                metrics = execute(model, data, initial, steps, output_dir=out,
                                  control_hz=FPS, on_frame=record)
            require_frames = round(data.time*FPS)
            if len(states) != require_frames:
                raise RuntimeError(f'Expected {require_frames} frames, got {len(states)}')
            rgb = np.stack(images)
            arrays = {'observation_state': np.stack(states), 'action': np.stack(actions),
                      'observation_images_overhead': rgb,
                      'timestamp': np.asarray(timestamps, dtype=np.float64),
                      'stage': np.asarray(stages, dtype=np.int16),
                      'active_arm': np.asarray(owners, dtype=np.int8),
                      'initial_qpos': initial_qpos, 'initial_qvel': initial_qvel,
                      'final_qpos': data.qpos.copy()}
            for key in ('observation_state', 'action'):
                if arrays[key].shape != (len(states), 12) or not np.isfinite(arrays[key]).all():
                    raise RuntimeError(f'Invalid {key}')
            if not np.allclose(np.diff(arrays['timestamp']), 1/FPS, atol=1e-8):
                raise RuntimeError('Nonuniform recording timestamps')
            # Only validated, complete episodes get a training archive.
            np.savez_compressed(out/'episode.npz', **arrays)
            write_png(out/'preview.png', rgb[len(rgb)//4])
            metadata.update(success=True, frames=len(states), duration_s=float(data.time), metrics=metrics,
                            archive_sha256=digest(out/'episode.npz'))
        except (RuntimeError, AssertionError, ValueError) as error:
            metadata['failure'] = f'{type(error).__name__}: {error}'
            metadata['frames_before_failure'] = len(states)
            print(metadata['failure'])
    metadata['wall_seconds'] = time.perf_counter()-started
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'data'/'relay_pilot')
    parser.add_argument('--train-episodes', type=int, default=2)
    parser.add_argument('--val-episodes', type=int, default=1)
    parser.add_argument('--train-start-seed', type=int, default=1000)
    parser.add_argument('--val-start-seed', type=int, default=10000)
    parser.add_argument('--initial-jitter', type=float, default=0., help='Initial non-gripper joint offset range in radians')
    parser.add_argument('--base-dataset', type=Path, help='Copy an existing raw dataset before appending corrective episodes')
    args = parser.parse_args()
    if not 1 <= args.train_episodes <= 1000 or not 1 <= args.val_episodes <= 1000:
        parser.error('Episode counts must be between 1 and 1000')
    if not 0 <= args.initial_jitter <= .06: parser.error('Initial jitter must be between 0 and 0.06 radians')
    base = json.loads((args.base_dataset/'manifest.json').read_text()) if args.base_dataset else None
    seeds = [e['seed'] for e in base['episodes']] if base else []
    seeds += list(range(args.train_start_seed,args.train_start_seed+args.train_episodes))
    seeds += list(range(args.val_start_seed,args.val_start_seed+args.val_episodes))
    if len(seeds) != len(set(seeds)): parser.error('Episode seeds must be unique across the base, train and validation splits')
    args.output.mkdir(parents=True, exist_ok=False)
    sources = ['relay_demo.py', 'grasp_demo.py', 'place_demo.py', 'reach_demo.py', 'dual_arm_demo.py', 'collect_demonstrations.py']
    manifest = {'format': 'tablesync-raw-v1', 'task': TASK, 'fps': FPS,
        'action_semantics': '12 absolute joint position targets in radians, held for 20 ms; observation precedes action',
        'joint_order': [arm+'_'+joint for arm in ('A', 'B') for joint in JOINTS],
        'policy_inputs': ['observation_state', 'observation_images_overhead'],
        'privileged_annotations_not_policy_inputs': ['stage', 'active_arm', 'initial_qpos', 'initial_qvel', 'final_qpos'],
        'render_flags': {'shadows': False, 'reflections': False},
        'python_version': platform.python_version(), 'mujoco_version': mujoco.__version__,
        'numpy_version': np.__version__, 'source_sha256': {s: digest(ROOT/'scripts'/s) for s in sources},
        'initial_jitter_range_rad': args.initial_jitter,
        'base_dataset': str(args.base_dataset.resolve()) if args.base_dataset else None,
        'episodes': list(base['episodes']) if base else []}
    if base:
        for split in ('train','val'): shutil.copytree(args.base_dataset/split,args.output/split)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for split, first, count in [('train', args.train_start_seed, args.train_episodes), ('val', args.val_start_seed, args.val_episodes)]:
        for seed in range(first, first+count):
            print(f'Recording {split} seed {seed} ...', flush=True)
            result = collect(seed, split, args.output, args.initial_jitter); manifest['episodes'].append(result)
            (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            print(f"  {'SAVED' if result['success'] else 'REJECTED'}: {result.get('frames', 0)} frames", flush=True)
    good = sum(e['success'] for e in manifest['episodes'])
    print(f'{good}/{len(manifest["episodes"])} episodes saved. Dataset: {args.output}')
    raise SystemExit(0 if good == len(manifest['episodes']) else 1)

if __name__ == '__main__': main()
