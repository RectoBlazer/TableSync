"""Validate raw archives and replay saved actions without the expert controller."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from collect_demonstrations import FPS, digest
from relay_demo import make_model, ROOT, support, FINAL
from reach_demo import contacts

def check(condition, message):
    if not condition: raise RuntimeError(message)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path, nargs='?', default=ROOT/'data'/'relay_pilot')
    args = parser.parse_args()
    manifest = json.loads((args.dataset/'manifest.json').read_text())
    seen = set(); results = []
    for episode in manifest['episodes']:
        seed = episode['seed']; split = episode['split']
        check(seed not in seen, 'Seed reused across episodes or splits'); seen.add(seed)
        check(episode['success'], f'Seed {seed} is a rejected episode')
        root = args.dataset/split/f'seed_{seed:06d}'
        path = root/'episode.npz'
        check(digest(path) == episode['archive_sha256'], f'Seed {seed}: archive hash mismatch')
        with np.load(path, allow_pickle=False) as archive:
            state = archive['observation_state']; action = archive['action']; times = archive['timestamp']
            rgb = archive['observation_images_overhead']; n = len(state)
            check(n == 3000 and action.shape == state.shape == (n, 12), 'Incorrect episode shape')
            check(rgb.shape == (n, 224, 224, 3) and rgb.dtype == np.uint8, 'Incorrect RGB format')
            check(np.std(rgb[0]) > 10 and not np.array_equal(rgb[0], rgb[n//4]), 'Blank or static RGB')
            del rgb
            check(np.isfinite(state).all() and np.isfinite(action).all(), 'Non-finite signals')
            check(np.allclose(times, np.arange(n)/FPS, atol=1e-8), 'Timestamp misalignment')
            model = make_model(seed); data = mujoco.MjData(model)
            data.qpos[:] = archive['initial_qpos']; data.qvel[:] = archive['initial_qvel']
            mujoco.mj_forward(model, data)
            stride = round(1/(FPS*model.opt.timestep)); max_state_error = 0.
            check(np.all(action >= model.actuator_ctrlrange[:, 0]-1e-6) and
                  np.all(action <= model.actuator_ctrlrange[:, 1]+1e-6), 'Action outside actuator limits')
            for i, target in enumerate(action):
                max_state_error = max(max_state_error, float(np.max(np.abs(data.qpos[:12]-state[i]))))
                data.ctrl[:] = target
                for _ in range(stride):
                    mujoco.mj_step(model, data)
                    check(contacts(model, data) == (0, 0), 'Replay collision')
            mujoco.mj_forward(model, data)
            check(max_state_error < .02, f'Replay state mismatch: {max_state_error} rad')
            block_q = model.jnt_qposadr[model.joint('block_free').id]
            error = float(np.linalg.norm(data.qpos[block_q:block_q+3]-archive['final_qpos'][block_q:block_q+3]))
            check(error < .005, f'Replay object mismatch: {error*1000} mm')
            check(support(model, data) == {'table'}, 'Replay did not release block')
            check(np.linalg.norm(data.body('block').xpos[:2]-FINAL) < .015, 'Replay missed final target')
            result = {'seed': seed, 'split': split, 'frames': n, 'max_state_error_rad': max_state_error,
                      'final_object_replay_error_mm': error*1000}
            results.append(result)
            print(f'PASS {split} seed {seed}: {n} aligned frames; replay error {error*1000:.3f} mm', flush=True)
    check(any(r['split'] == 'train' for r in results) and any(r['split'] == 'val' for r in results), 'Missing split')
    report = {'success': True, 'total_frames': sum(r['frames'] for r in results), 'episodes': results}
    (args.dataset/'validation.json').write_text(json.dumps(report, indent=2))
    print(f"PASS dataset: {report['total_frames']} frames, independent train/validation seeds.")

if __name__ == '__main__': main()
