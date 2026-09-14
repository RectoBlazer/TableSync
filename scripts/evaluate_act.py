"""Closed-loop ACT evaluation. Simulator state is used only to score/stop runs."""
import argparse
import hashlib
import training_runtime
import json
from pathlib import Path
import time
import mujoco
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from relay_demo import make_model, HOME, support, DESTINATION, FINAL
from grasp_demo import plan as grasp_plan
from reach_demo import contacts
from dual_arm_demo import ROOT, write_png

class RelayMonitor:
    """Ordered physical events: a final-position-only check would reward doing nothing."""
    def __init__(self):
        self.events = {}; self.consecutive = 0

    def update(self, position, surfaces, time_s):
        if 'A_lift' not in self.events:
            condition = position[2] > .05 and {'A_fixed','A_moving'} <= surfaces
            event, needed = 'A_lift', 10
        elif 'A_transfer' not in self.events:
            condition = surfaces == {'table'} and np.linalg.norm(position[:2]-DESTINATION) < .015
            event, needed = 'A_transfer', 10
        elif 'B_lift' not in self.events:
            condition = position[2] > .05 and {'B_fixed','B_moving'} <= surfaces
            event, needed = 'B_lift', 10
        else:
            condition = surfaces == {'table'} and np.linalg.norm(position[:2]-FINAL) < .015
            event, needed = 'B_final', 25
        self.consecutive = self.consecutive+1 if condition else 0
        if self.consecutive >= needed and event not in self.events:
            self.events[event] = time_s; self.consecutive = 0
            print(f'  Verified {event} at {time_s:.2f}s', flush=True)

def run(seed, policy, pre, post, output, view_enabled=False, execute_actions=10, policy_input_names=None):
    folder = output/f'seed_{seed:06d}'; folder.mkdir(parents=True, exist_ok=False)
    model = make_model(seed)
    # The same fixed pregrasp initialization as recording, not an expert rollout.
    initial, _ = grasp_plan(model)
    data = mujoco.MjData(model); data.qpos[:6] = initial; data.qpos[6:12] = HOME
    data.ctrl[:] = data.qpos[:12]; mujoco.mj_forward(model, data)
    policy.reset(); monitor = RelayMonitor(); latencies = []; trace = []; actions = []; states = []
    report = {'seed': seed, 'success': False, 'policy_inputs': ['RGB', '12 joint angles'],
              'action_hz': 50, 'inference_hz': 50/execute_actions, 'execute_actions': execute_actions,
              'events': monitor.events, 'clipped_action_values': 0}
    task_time = policy.config.input_features['observation.state'].shape[0] == 13
    if task_time: report['policy_inputs'].append('elapsed episode time / 60')
    if policy_input_names is not None: report['policy_inputs'] = policy_input_names
    prior = getattr(policy, 'trajectory_prior', None)
    if prior is not None:
        report['controller'] = 'mean training trajectory plus learned ACT residual'
        report['max_abs_learned_residual_rad'] = 0.
    camera = mujoco.MjvCamera(); camera.lookat[:] = [-.08,0,.1]
    camera.distance = .95; camera.azimuth = 90; camera.elevation = -35
    view = None; last_image = None; chunk = None
    if view_enabled:
        from mujoco import viewer
        view = viewer.launch_passive(model, data)
        view.cam.lookat[:] = camera.lookat; view.cam.distance = camera.distance
        view.cam.azimuth = camera.azimuth; view.cam.elevation = camera.elevation
    started = time.perf_counter()
    try:
        with mujoco.Renderer(model, height=224, width=224) as renderer:
            for frame in range(3000):
                wall = time.perf_counter()
                mujoco.mj_forward(model, data)
                if frame % execute_actions == 0:
                    renderer.update_scene(data, camera=camera)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                    last_image = renderer.render().copy()
                    state = np.r_[data.qpos[:12], data.time/60.] if task_time else data.qpos[:12]
                    observation = {'observation.state': torch.from_numpy(state.astype(np.float32)),
                        'observation.images.overhead': torch.from_numpy(last_image.copy()).permute(2,0,1).float()/255.}
                    tick = time.perf_counter()
                    with torch.inference_mode():
                        batch = pre(observation)
                        chunk = post(policy.predict_action_chunk(batch))[0].cpu().numpy()
                    if prior is not None:
                        report['max_abs_learned_residual_rad'] = max(report['max_abs_learned_residual_rad'],float(np.max(np.abs(chunk))))
                        indices = np.minimum(np.arange(frame,frame+len(chunk)),len(prior)-1)
                        chunk = chunk + prior[indices]
                    latencies.append((time.perf_counter()-tick)*1000)
                    if chunk.shape != (policy.config.chunk_size,12) or not np.isfinite(chunk).all():
                        raise RuntimeError('Invalid policy action chunk')
                target = chunk[frame % execute_actions]
                clipped = np.clip(target, model.actuator_ctrlrange[:,0], model.actuator_ctrlrange[:,1])
                report['clipped_action_values'] += int(np.count_nonzero(clipped != target))
                states.append(data.qpos[:12].copy()); actions.append(clipped.copy())
                data.ctrl[:] = clipped
                for _ in range(10):
                    mujoco.mj_step(model, data)
                    if not np.isfinite(data.qpos).all(): raise RuntimeError('Non-finite physics state')
                    cross, table = contacts(model, data)
                    if cross or table:
                        report['collision_contacts'] = []
                        for contact in data.contact:
                            if contact.dist >= 0: continue
                            geoms = [model.geom(g).name for g in (contact.geom1, contact.geom2)]
                            bodies = [model.body(int(model.geom_bodyid[g])).name for g in (contact.geom1, contact.geom2)]
                            monitored = ('table' in geoms and any(n.startswith(('A_', 'B_')) for n in bodies)) or (any(n.startswith('A_') for n in bodies) and any(n.startswith('B_') for n in bodies))
                            if monitored:
                                report['collision_contacts'].append({'bodies': bodies, 'geoms': geoms, 'penetration_m': -float(contact.dist)})
                        raise RuntimeError(f'Collision detected: inter-arm={cross}, arm-table={table}')
                mujoco.mj_forward(model, data)
                position = data.body('block').xpos.copy(); surfaces = support(model, data)
                trace.append(np.r_[data.time, position])
                monitor.update(position, surfaces, float(data.time))
                if position[2] < -.02: raise RuntimeError('Object fell off table')
                if view is not None:
                    if not view.is_running(): raise RuntimeError('Viewer closed before completion')
                    view.sync(); time.sleep(max(0,.02-(time.perf_counter()-wall)))
                if 'B_final' in monitor.events:
                    report['success'] = True; break
            if not report['success']:
                report['failure'] = 'Time limit; missing ordered milestones: '+', '.join(
                    e for e in ('A_lift','A_transfer','B_lift','B_final') if e not in monitor.events)
    except RuntimeError as error:
        report['failure'] = str(error)
    finally:
        if view is not None: view.close()
    report.update(simulated_seconds=float(data.time), wall_seconds=time.perf_counter()-started,
                  final_object_position_m=data.body('block').xpos.tolist())
    if trace: report['peak_block_height_m'] = float(np.max(np.asarray(trace)[:,3]))
    if latencies:
        report['inference_calls'] = len(latencies)
        report['inference_median_ms'] = float(np.median(latencies))
        report['inference_p95_ms'] = float(np.percentile(latencies,95))
    if last_image is not None: write_png(folder/'last_observation.png', last_image)
    np.savez_compressed(folder/'trajectory.npz', states=np.asarray(states), actions=np.asarray(actions),
                        object_trace=np.asarray(trace))
    (folder/'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'outputs/act_pilot/pretrained_model')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/act_closed_loop_pilot')
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--execute-actions', type=int, default=10, help='Actions consumed before replanning; default timing is unchanged')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds): parser.error('Seeds must be unique')
    args.output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(4)
    policy = ACTPolicy.from_pretrained(args.checkpoint).cuda().eval()
    metadata_path = args.checkpoint/'controller.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata['type'] == 'trajectory_prior_act_residual':
            prior_path = args.checkpoint/'trajectory_prior.npy'
            if not prior_path.exists() or hashlib.sha256(prior_path.read_bytes()).hexdigest() != metadata['prior_sha256']:
                raise ValueError('Required trajectory reference is missing or has changed')
    if (args.checkpoint/'trajectory_prior.npy').exists():
        policy.trajectory_prior = np.load(args.checkpoint/'trajectory_prior.npy')
        if policy.trajectory_prior.shape != (3000,12) or not np.isfinite(policy.trajectory_prior).all():
            raise ValueError('Invalid training trajectory reference')
    if not 1 <= args.execute_actions <= policy.config.chunk_size:
        raise ValueError('Execution length must fit within the policy action chunk')
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint))
    report = {'checkpoint': str(args.checkpoint.resolve()), 'mujoco_version': mujoco.__version__,
              'torch_version': torch.__version__, 'numpy_version': np.__version__, 'trials': []}
    for seed in args.seeds:
        print(f'ACT closed-loop seed {seed}', flush=True)
        report['trials'].append(run(seed, policy, pre, post, args.output, args.view, args.execute_actions))
        report['successes'] = sum(r['success'] for r in report['trials'])
        (args.output/'summary.json').write_text(json.dumps(report, indent=2))
    print(f"Completed: {report['successes']}/{len(args.seeds)} successful.")
    raise SystemExit(0 if report['successes'] == len(args.seeds) else 1)

if __name__ == '__main__': main()
