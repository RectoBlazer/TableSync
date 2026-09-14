"""Small local ACT baseline using official LeRobot datasets, policy and processors."""
import argparse
import copy
import hashlib
import training_runtime  # Configure caches before importing Hugging Face libraries.
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from lerobot.configs.types import FeatureType
from fast_lerobot_dataset import CachedActionDataset as LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

ROOT = Path(__file__).resolve().parents[1]

def save_controller_metadata(folder, prior, task_time):
    metadata = {'type': 'trajectory_prior_act_residual' if prior is not None else 'absolute_act',
                'task_time_input': task_time, 'control_hz': 50, 'episode_seconds': 60}
    if prior is not None:
        np.save(folder/'trajectory_prior.npy', prior)
        metadata['prior_sha256'] = hashlib.sha256((folder/'trajectory_prior.npy').read_bytes()).hexdigest()
    (folder/'controller.json').write_text(json.dumps(metadata, indent=2))

@torch.no_grad()
def evaluate(policy, preprocessor, dataset, stats):
    policy.eval(); errors = []; normalized = []
    indices = list(range(0, len(dataset), 100))
    scale = torch.as_tensor(stats['action']['std'], device='cuda', dtype=torch.float32)
    for raw in DataLoader(Subset(dataset, indices), batch_size=4, num_workers=0):
        batch = preprocessor(raw)
        inputs = {k: v for k, v in batch.items() if k.startswith('observation.')}
        prediction = policy.predict_action_chunk(inputs)
        valid = (~batch['action_is_pad']).unsqueeze(-1).expand_as(prediction)
        difference = (prediction-batch['action']).abs()
        normalized.append(difference[valid].cpu())
        errors.append((difference*scale)[valid].cpu())
    return {'sampled_frames': len(indices), 'action_chunk_mae_rad': float(torch.cat(errors).mean()),
            'normalized_action_chunk_mae': float(torch.cat(normalized).mean())}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/lerobot_relay_pilot')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/act_pilot')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, help='Override learning rate for fine-tuning')
    parser.add_argument('--task-time', action='store_true', help='Append observable elapsed episode time / 60 to joint observations')
    parser.add_argument('--trajectory-prior', action='store_true', help='Train ACT residuals around a mean training trajectory; requires --task-time')
    parser.add_argument('--save-every', type=int, default=0, help='Save intermediate policy/processor checkpoints at this interval')
    parser.add_argument('--init-checkpoint', type=Path, help='Warm-start model weights; creates a fresh optimizer')
    parser.add_argument('--grasp-weight', type=float, default=1., help='Sampling weight for approach/close frames')
    parser.add_argument('--sampling-stages', nargs='+', default=['approach','close'])
    parser.add_argument('--idle-weight', type=float, default=1., help='Sampling multiplier for nearly unchanged action targets over the next second')
    parser.add_argument('--raw-dataset', type=Path, default=ROOT/'data/relay_pilot')
    args = parser.parse_args()
    if args.trajectory_prior and not args.task_time: parser.error('--trajectory-prior requires --task-time')
    if args.init_checkpoint and (args.init_checkpoint/'trajectory_prior.npy').exists() and not args.trajectory_prior:
        parser.error('A residual checkpoint requires --trajectory-prior to preserve action semantics')
    if args.steps < 1 or args.batch_size < 1: parser.error('Steps and batch size must be positive')
    if args.grasp_weight < 1: parser.error('Grasp weight must be at least 1')
    if not 0 < args.idle_weight <= 1: parser.error('Idle weight must be in (0,1]')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable in the training environment')
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    delta = {'action': [i/50 for i in range(50)]}
    train = LeRobotDataset('local/tablesync_train', root=args.dataset/'train', delta_timestamps=delta)
    val = LeRobotDataset('local/tablesync_val', root=args.dataset/'val', delta_timestamps=delta)
    features = dataset_to_policy_features(train.features)
    stats = copy.deepcopy(train.meta.stats)
    prior = None
    if args.trajectory_prior:
        manifest = json.loads((args.raw_dataset/'manifest.json').read_text())
        export = json.loads((args.dataset/'export_report.json').read_text())
        episodes = [e for e in manifest['episodes'] if e['split'] == 'train' and e['success']]
        if [e['seed'] for e in episodes] != export['train']['seeds']:
            raise ValueError('Trajectory prior training episodes do not match export')
        action_arrays = []
        for ep in episodes:
            with np.load(args.raw_dataset/'train'/f"seed_{ep['seed']:06d}"/'episode.npz') as raw:
                action_arrays.append(raw['action'].copy())
        actions = np.stack(action_arrays)
        prior = actions.mean(axis=0).astype(np.float32)
        residuals = (actions-prior).reshape(-1,12)
        stats['action'] = {'mean': residuals.mean(axis=0),
            'std': np.maximum(residuals.std(axis=0), .02),
            'min': residuals.min(axis=0), 'max': residuals.max(axis=0)}
        train.action_prior = val.action_prior = torch.from_numpy(prior)
    if args.task_time:
        train.include_task_time = val.include_task_time = True
        features['observation.state'].shape = (13,)
        # Fixed scaling for a clock in [0,1], independent of validation data.
        for key, value in list(stats['observation.state'].items()):
            if key == 'count': continue
            appended = {'mean': .5, 'std': 1/np.sqrt(12), 'min': 0., 'max': 1.}.get(key)
            if appended is not None:
                stats['observation.state'][key] = np.r_[np.asarray(value), appended].astype(np.float32)
    cfg = ACTConfig(input_features={k:v for k,v in features.items() if v.type != FeatureType.ACTION},
        output_features={k:v for k,v in features.items() if v.type == FeatureType.ACTION},
        device='cuda', chunk_size=50, n_action_steps=10, dim_model=256, n_heads=4,
        dim_feedforward=1024, n_encoder_layers=2, n_decoder_layers=1, n_vae_encoder_layers=2,
        optimizer_lr=1e-4, optimizer_lr_backbone=1e-5, push_to_hub=False)
    if args.init_checkpoint:
        policy = ACTPolicy.from_pretrained(args.init_checkpoint).cuda()
        cfg = policy.config
        if cfg.chunk_size != 50: raise RuntimeError('This training loop expects 50-action chunks')
        if args.task_time and cfg.input_features['observation.state'].shape[0] == 12:
            cfg.input_features['observation.state'].shape = (13,)
            for name in ('encoder_robot_state_input_proj', 'vae_encoder_robot_state_input_proj'):
                old = getattr(policy.model, name, None)
                if old is None: continue
                new = torch.nn.Linear(13, old.out_features).cuda()
                with torch.no_grad():
                    new.weight[:, :12].copy_(old.weight); new.weight[:, 12].zero_()
                    new.bias.copy_(old.bias)
                setattr(policy.model, name, new)
        if cfg.input_features['observation.state'].shape[0] != (13 if args.task_time else 12):
            raise ValueError('Checkpoint observation size does not match --task-time')
    else:
        policy = ACTPolicy(cfg).cuda()
    if args.trajectory_prior:
        if args.init_checkpoint and (args.init_checkpoint/'trajectory_prior.npy').exists():
            old_prior = np.load(args.init_checkpoint/'trajectory_prior.npy')
            if not np.array_equal(prior, old_prior): raise ValueError('Warm-start trajectory prior differs')
        else:
            # Start at the training reference; learn corrections rather than relearn its motion.
            torch.nn.init.zeros_(policy.model.action_head.weight)
            torch.nn.init.zeros_(policy.model.action_head.bias)
    if args.lr is not None:
        if args.lr <= 0: raise ValueError('Learning rate must be positive')
        cfg.optimizer_lr = args.lr
        cfg.optimizer_lr_backbone = min(args.lr,1e-5)
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=cfg.optimizer_lr, weight_decay=1e-4)
    sampler = None
    if args.grasp_weight > 1 or args.idle_weight < 1:
        manifest = json.loads((args.raw_dataset/'manifest.json').read_text())
        export = json.loads((args.dataset/'export_report.json').read_text())
        raw_seeds = [e['seed'] for e in manifest['episodes'] if e['split'] == 'train' and e['success']]
        if raw_seeds != export['train']['seeds']:
            raise RuntimeError('Raw stage annotations and LeRobot episode order differ')
        weights = []
        for ep in manifest['episodes']:
            if ep['split'] != 'train' or not ep['success']: continue
            folder = args.raw_dataset/'train'/f"seed_{ep['seed']:06d}"
            metadata = json.loads((folder/'metadata.json').read_text())
            selected = [metadata['stage_names'].index(n) for n in args.sampling_stages]
            with np.load(folder/'episode.npz',allow_pickle=False) as raw:
                phase_weights = np.where(np.isin(raw['stage'],selected),args.grasp_weight,1.)
                actions = raw['action']
                future = actions[np.minimum(np.arange(len(actions))+50,len(actions)-1)]
                idle = ((np.max(np.abs(future-actions),axis=1) < .002) &
                        (np.max(np.abs(actions-raw['observation_state']),axis=1) < .01))
                phase_weights[idle] *= args.idle_weight
                weights.extend(phase_weights.tolist())
        if len(weights) != len(train): raise RuntimeError('Raw annotations do not align with the training export')
        sampler = WeightedRandomSampler(weights,len(weights),replacement=True)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=sampler is None,
                        sampler=sampler, num_workers=0, drop_last=True)
    report = {'steps': args.steps, 'batch_size': args.batch_size, 'seed': 42,
        'task_time_input': args.task_time,
        'trajectory_prior': args.trajectory_prior,
        'init_checkpoint': str(args.init_checkpoint) if args.init_checkpoint else None,
        'grasp_sampling_weight': args.grasp_weight,
        'sampling_stages': args.sampling_stages,
        'idle_sampling_weight': args.idle_weight,
        'learning_rate': cfg.optimizer_lr,
        'device': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
        'train_frames': len(train), 'validation_frames': len(val),
        'parameters': sum(p.numel() for p in policy.parameters()),
        'scope': 'offline pilot training; not closed-loop task success', 'training': []}
    report['validation_before'] = evaluate(policy, pre, val, stats)
    print('Validation before:', report['validation_before'], flush=True)
    start = time.perf_counter(); iteration = iter(loader)
    for step in range(1, args.steps+1):
        try: raw = next(iteration)
        except StopIteration: iteration = iter(loader); raw = next(iteration)
        policy.train(); batch = pre(raw)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = policy(batch)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
        optimizer.step()
        if args.save_every > 0 and step % args.save_every == 0:
            intermediate = args.output/'checkpoints'/f'step_{step:06d}'
            policy.save_pretrained(intermediate); pre.save_pretrained(intermediate); post.save_pretrained(intermediate)
            save_controller_metadata(intermediate, prior, args.task_time)
            print(f'CHECKPOINT READY: {intermediate}', flush=True)
        if step == 1 or step % 25 == 0 or step == args.steps:
            row = {'step': step, 'loss': float(loss.detach()), **metrics}
            report['training'].append(row); print(row, flush=True)
            (args.output/'progress.json').write_text(json.dumps(report, indent=2))
    torch.cuda.synchronize(); report['training_seconds'] = time.perf_counter()-start
    report['validation_after'] = evaluate(policy, pre, val, stats)
    checkpoint = args.output/'pretrained_model'
    policy.save_pretrained(checkpoint); pre.save_pretrained(checkpoint); post.save_pretrained(checkpoint)
    save_controller_metadata(checkpoint, prior, args.task_time)
    # Optimizer/RNG state supports a future resume implementation, separate from inference assets.
    torch.save({'step': args.steps, 'optimizer': optimizer.state_dict(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, args.output/'training_state.pt')
    reloaded = ACTPolicy.from_pretrained(checkpoint).cuda().eval()
    reload_pre, reload_post = make_pre_post_processors(cfg, pretrained_path=str(checkpoint))
    sample = pre(next(iter(DataLoader(val, batch_size=1))))
    inputs = {k:v for k,v in sample.items() if k.startswith('observation.')}
    with torch.no_grad():
        difference = (policy.predict_action_chunk(inputs)-reloaded.predict_action_chunk(inputs)).abs().max().item()
    if difference > 1e-6: raise RuntimeError('Reloaded checkpoint predictions differ')
    raw_check = next(iter(DataLoader(val, batch_size=1)))
    normalized_check = reload_pre(raw_check)
    reload_inputs = {k:v for k,v in normalized_check.items() if k.startswith('observation.')}
    physical_actions = reload_post(reloaded.predict_action_chunk(reload_inputs))
    if physical_actions.shape != (1,50,12) or not torch.isfinite(physical_actions).all():
        raise RuntimeError('Reloaded processors produced invalid actions')
    report['checkpoint_reload_max_difference'] = difference
    report['peak_gpu_memory_mb'] = torch.cuda.max_memory_allocated()/1024**2
    (args.output/'training_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)); print(f'SAVED: {checkpoint}')

if __name__ == '__main__': main()
