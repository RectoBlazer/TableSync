"""Teacher-forced action errors by expert stage, for diagnosing physical failures."""
import argparse
import training_runtime
import json
from pathlib import Path
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from training_runtime import ROOT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--episode', type=Path, default=ROOT/'data/relay_pilot/train/seed_001000')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(4)
    policy = ACTPolicy.from_pretrained(args.checkpoint).cuda().eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint))
    meta = json.loads((args.episode/'metadata.json').read_text())
    results = {}
    with np.load(args.episode/'episode.npz', allow_pickle=False) as raw:
        images = raw['observation_images_overhead']; states = raw['observation_state']
        actions = raw['action']; stages = raw['stage']; arms = raw['active_arm']
        for i in range(0,len(states),10):
            observation = {'observation.state': torch.from_numpy(states[i].copy()),
                'observation.images.overhead': torch.from_numpy(images[i].copy()).permute(2,0,1).float()/255.}
            with torch.inference_mode(): prediction = post(policy.predict_action_chunk(pre(observation)))[0].cpu().numpy()
            n = min(10,len(actions)-i)
            error = np.abs(prediction[:n]-actions[i:i+n])
            name = ('B' if arms[i] else 'A')+'/'+meta['stage_names'][stages[i]]
            results.setdefault(name,[]).append(error.mean(axis=0))
    summary = {k: {'mean_action_error_rad': float(np.mean(v)),
                   'per_joint_mae_rad': np.mean(v,axis=0).tolist()} for k,v in results.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2))
    for k,v in summary.items(): print(k,round(v['mean_action_error_rad'],5),flush=True)

if __name__ == '__main__': main()
