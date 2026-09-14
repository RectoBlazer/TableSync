"""Export validated raw episodes through the official LeRobot v3 writer, locally."""
import argparse
import training_runtime  # Configure caches before importing Hugging Face libraries.
import hashlib
import json
from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parents[1]

def verify_export(source, output):
    manifest = json.loads((source/'manifest.json').read_text())
    checked = 0
    for split in ('train', 'val'):
        ds = LeRobotDataset(f'local/tablesync_{split}', root=output/split)
        chunks = LeRobotDataset(f'local/tablesync_{split}', root=output/split,
                               delta_timestamps={'action': [0., .02, 1.]})
        offset = 0
        for ep in manifest['episodes']:
            if ep['split'] != split or not ep['success']: continue
            with np.load(source/split/f"seed_{ep['seed']:06d}"/'episode.npz', allow_pickle=False) as raw:
                state = raw['observation_state']; actions = raw['action']; rgb = raw['observation_images_overhead']
                for i in (0, len(state)-1):
                    sample = ds[offset+i]
                    np.testing.assert_allclose(sample['observation.state'].numpy(), state[i], atol=1e-7)
                    np.testing.assert_allclose(sample['action'].numpy(), actions[i], atol=1e-7)
                    np.testing.assert_allclose(sample['observation.images.overhead'].numpy(),
                                               rgb[i].transpose(2,0,1)/255., atol=1e-7)
                    checked += 1
                last = chunks[offset+len(state)-1]
                if last['action_is_pad'].tolist() != [False, True, True]:
                    raise RuntimeError('Action chunk crosses an episode boundary')
                offset += len(state)
    result = {'success': True, 'boundary_frames_compared_to_raw': checked,
              'checks': ['joint values', 'action values', 'RGB pixels', 'episode chunk padding']}
    (output/'integrity_report.json').write_text(json.dumps(result, indent=2))
    print('PASS export integrity:', result, flush=True)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT/'data/relay_pilot')
    parser.add_argument('--output', type=Path, default=ROOT/'data/lerobot_relay_pilot')
    parser.add_argument('--resume', action='store_true', help='Reuse complete split exports after an interrupted run')
    args = parser.parse_args()
    manifest = json.loads((args.source/'manifest.json').read_text())
    validation = json.loads((args.source/'validation.json').read_text())
    if not validation['success']: raise RuntimeError('Raw dataset did not pass validation')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    features = {
        'observation.state': {'dtype': 'float32', 'shape': (12,), 'names': manifest['joint_order']},
        'action': {'dtype': 'float32', 'shape': (12,), 'names': manifest['joint_order']},
        'observation.images.overhead': {'dtype': 'image', 'shape': (224,224,3), 'names': ['height','width','channels']}}
    results = {}
    for split in ('train', 'val'):
        expected = [e for e in manifest['episodes'] if e['split'] == split and e['success']]
        if args.resume and (args.output/split).exists():
            loaded = LeRobotDataset(f'local/tablesync_{split}', root=args.output/split)
            if len(loaded) != sum(e['frames'] for e in expected):
                raise RuntimeError('Incomplete split; choose a fresh output directory')
            results[split] = {'seeds': [e['seed'] for e in expected], 'frames': len(loaded), 'episodes': len(expected)}
            print(f'Reused complete {split} export', flush=True)
            continue
        ds = LeRobotDataset.create(repo_id=f'local/tablesync_{split}', fps=manifest['fps'],
            root=args.output/split, robot_type='dual_so101_mujoco', features=features,
            use_videos=False, image_writer_threads=4)
        seeds = []; count = 0
        try:
            for ep in manifest['episodes']:
                if ep['split'] != split or not ep['success']: continue
                seed = ep['seed']; path = args.source/split/f'seed_{seed:06d}'/'episode.npz'
                if hashlib.sha256(path.read_bytes()).hexdigest() != ep['archive_sha256']:
                    raise RuntimeError(f'Raw archive changed for seed {seed}')
                with np.load(path, allow_pickle=False) as raw:
                    state = raw['observation_state']; actions = raw['action']; rgb = raw['observation_images_overhead']
                    for i in range(len(state)):
                        ds.add_frame({'observation.state': state[i], 'action': actions[i],
                                      'observation.images.overhead': rgb[i], 'task': manifest['task']})
                    ds.save_episode(); count += len(state); seeds.append(seed)
                print(f'Exported {split} seed {seed}: {len(state)} frames', flush=True)
        finally:
            ds.finalize()
        loaded = LeRobotDataset(repo_id=f'local/tablesync_{split}', root=args.output/split)
        if len(loaded) != count: raise RuntimeError('Export frame count mismatch')
        sample = loaded[0]
        if tuple(sample['observation.images.overhead'].shape) != (3,224,224):
            raise RuntimeError('Unexpected decoded image shape')
        results[split] = {'seeds': seeds, 'frames': count, 'episodes': len(seeds)}
    if set(results['train']['seeds']) & set(results['val']['seeds']): raise RuntimeError('Split leakage')
    (args.output/'export_report.json').write_text(json.dumps(results, indent=2))
    verify_export(args.source, args.output)
    print(json.dumps(results, indent=2)); print('PASS: local LeRobot datasets written and reloaded.')

if __name__ == '__main__': main()
