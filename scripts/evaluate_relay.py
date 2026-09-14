"""Reproducible, modest domain randomization for the scripted table relay."""
import argparse
from contextlib import redirect_stdout
import json
import platform
import time
import mujoco
import numpy as np
from relay_demo import make_model, plan, execute, HOME, ROOT

def run_trial(seed, root):
    out = root/f'seed_{seed:03d}'; out.mkdir(parents=True, exist_ok=True)
    result = {'seed': seed, 'success': False}
    started = time.perf_counter()
    model = data = None
    with (out/'run.log').open('w', encoding='utf-8') as log, redirect_stdout(log):
        try:
            model = make_model(seed)
            obj = model.body('block').id; geom = model.geom('block_geom').id
            result['parameters'] = {
                'initial_position_m': model.body_pos[obj].tolist(),
                'mass_kg': float(model.body_mass[obj]),
                'sliding_friction': float(model.geom_friction[geom, 0])}
            initial, steps = plan(model)
            data = mujoco.MjData(model)
            data.qpos[:6] = initial; data.qpos[6:12] = HOME
            data.ctrl[:] = data.qpos[:12]; mujoco.mj_forward(model, data)
            result['metrics'] = execute(model, data, initial, steps, output_dir=out)
            result['success'] = True
        except (RuntimeError, AssertionError, ValueError) as error:
            result['failure'] = f'{type(error).__name__}: {error}'
            print(result['failure'])
        if data is not None:
            result['simulated_seconds'] = float(data.time)
            result['last_object_position_m'] = data.body('block').xpos.tolist()
    result['wall_seconds'] = time.perf_counter()-started
    (out/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=10, help='Number of consecutive seeds')
    parser.add_argument('--start-seed', type=int, default=0)
    args = parser.parse_args()
    if args.seeds < 1: parser.error('--seeds must be positive')
    root = ROOT/'artifacts'/f'relay_eval_{args.start_seed}_{args.seeds}'
    root.mkdir(parents=True, exist_ok=True)
    report = {'task': 'scripted table-supported relay; simulator state feedback',
        'mujoco_version': mujoco.__version__, 'numpy_version': np.__version__,
        'python_version': platform.python_version(),
        'randomization': {'xy_offset_mm': [-3, 3], 'mass_g': [30, 50], 'sliding_friction': [.8, 1.2]},
        'requested_trials': args.seeds, 'trials': []}
    for seed in range(args.start_seed, args.start_seed+args.seeds):
        result = run_trial(seed, root); report['trials'].append(result)
        passed = sum(t['success'] for t in report['trials'])
        report.update(completed_trials=len(report['trials']), successes=passed,
                      success_rate=passed/len(report['trials']))
        (root/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        detail = (f"final error {result['metrics']['arms']['B']['placement_error_mm']:.1f} mm"
                  if result['success'] else result['failure'])
        print(f"Seed {seed}: {'PASS' if result['success'] else 'FAIL'} | {detail}", flush=True)
    print(f"Completed: {passed}/{args.seeds} successful ({100*report['success_rate']:.0f}%).")
    print(f'Report: {root / "summary.json"}')
    raise SystemExit(0 if passed == args.seeds else 1)

if __name__ == '__main__': main()
