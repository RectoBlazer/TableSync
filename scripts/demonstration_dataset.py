"""Dependency-light reader for action-chunk training; no privileged state inputs."""
import bisect
import json
from pathlib import Path
import numpy as np

def training_statistics(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    values = {'observation_state': [], 'action': []}; seeds = []
    for ep in manifest['episodes']:
        if ep['split'] != 'train' or not ep['success']: continue
        seeds.append(ep['seed'])
        with np.load(root/'train'/f"seed_{ep['seed']:06d}"/'episode.npz', allow_pickle=False) as data:
            for key in values: values[key].append(data[key])
    if not seeds: raise ValueError('No successful training episodes')
    stats = {'training_seeds': seeds}
    for key, chunks in values.items():
        x = np.concatenate(chunks).astype(np.float64)
        stats[key] = {'mean': x.mean(axis=0).tolist(),
                      'std': np.maximum(x.std(axis=0), 1e-6).tolist()}
    (root/'training_stats.json').write_text(json.dumps(stats, indent=2))
    return stats

class DemonstrationDataset:
    """NumPy samples usable by a PyTorch DataLoader after torch is installed.

    Keeps one RGB episode in memory. Action chunks never cross episode boundaries.
    Padding repeats the last action, with a boolean mask identifying padded steps.
    """
    def __init__(self, root, split='train', chunk_size=100):
        self.root = Path(root)
        if split not in ('train', 'val') or chunk_size < 1: raise ValueError('Invalid split/chunk size')
        self.chunk_size = chunk_size
        manifest = json.loads((self.root/'manifest.json').read_text())
        self.episodes = [e for e in manifest['episodes'] if e['split'] == split and e['success']]
        if not self.episodes: raise ValueError(f'No successful {split} episodes')
        stats_path = self.root/'training_stats.json'
        self.stats = json.loads(stats_path.read_text()) if stats_path.exists() else training_statistics(self.root)
        self.ends = np.cumsum([e['frames'] for e in self.episodes]).tolist()
        self.cached_episode = None; self.cache = None

    def __len__(self): return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self): raise IndexError(index)
        episode = bisect.bisect_right(self.ends, index)
        frame = index-(0 if episode == 0 else self.ends[episode-1])
        if episode != self.cached_episode:
            e = self.episodes[episode]
            self.cache = None
            with np.load(self.root/e['split']/f"seed_{e['seed']:06d}"/'episode.npz', allow_pickle=False) as data:
                self.cache = {k: data[k] for k in ('observation_state', 'action', 'observation_images_overhead')}
            self.cached_episode = episode
        count = len(self.cache['action'])
        offsets = frame+np.arange(self.chunk_size)
        actions = self.cache['action'][np.minimum(offsets, count-1)].copy()
        state = self.cache['observation_state'][frame].copy()
        def normalize(x, key):
            stat = self.stats[key]
            return ((x-np.asarray(stat['mean']))/np.asarray(stat['std'])).astype(np.float32)
        return {'observation.state': normalize(state, 'observation_state'),
                'observation.images.overhead': self.cache['observation_images_overhead'][frame].transpose(2,0,1).astype(np.float32)/255.,
                'action': normalize(actions, 'action'), 'action_is_pad': offsets >= count}
