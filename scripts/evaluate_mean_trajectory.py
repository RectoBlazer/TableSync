"""Diagnostic only: replay the mean training trajectory, without visual feedback."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import training_runtime
import numpy as np
import torch
from evaluate_act import run

class MeanTrajectory:
    def __init__(self, actions):
        self.actions = torch.from_numpy(actions)
        self.config = SimpleNamespace(chunk_size=50, input_features={'observation.state': SimpleNamespace(shape=(13,))})
    def reset(self): pass
    def predict_action_chunk(self, batch):
        frame = round(float(batch['observation.state'][-1])*3000)
        indices = torch.arange(frame, frame+50).clamp(0,len(self.actions)-1)
        return self.actions[indices].unsqueeze(0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, default=Path('data/relay_corrective'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[1000,0,1,2])
    args = p.parse_args()
    manifest = json.loads((args.dataset/'manifest.json').read_text())
    training = [e for e in manifest['episodes'] if e['split']=='train' and e['success']]
    arrays = []
    for ep in training:
        with np.load(args.dataset/'train'/f"seed_{ep['seed']:06d}"/'episode.npz') as raw:
            arrays.append(raw['action'].copy())
    mean = np.mean(np.stack(arrays),axis=0).astype(np.float32)
    args.output.mkdir(parents=True,exist_ok=False)
    np.save(args.output/'mean_training_actions.npy',mean)
    policy = MeanTrajectory(mean)
    report = {'scope': 'open-loop mean training trajectory diagnostic; NOT a vision policy',
              'training_seeds': [e['seed'] for e in training], 'trials': []}
    for seed in args.seeds:
        report['trials'].append(run(seed,policy,lambda x:x,lambda x:x,args.output,
                                   policy_input_names=['elapsed episode time only']))
        report['successes'] = sum(r['success'] for r in report['trials'])
        (args.output/'summary.json').write_text(json.dumps(report,indent=2))

if __name__ == '__main__': main()
