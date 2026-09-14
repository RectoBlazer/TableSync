"""Replay saved policy commands to identify the first monitored collision."""
import argparse
import json
import mujoco
import numpy as np
from relay_demo import make_model, HOME
from grasp_demo import plan
from reach_demo import contacts

def main():
    p = argparse.ArgumentParser()
    p.add_argument('trajectory')
    p.add_argument('--seed', type=int, required=True)
    args = p.parse_args()
    model = make_model(args.seed)
    initial, _ = plan(model)
    data = mujoco.MjData(model)
    data.qpos[:6] = initial; data.qpos[6:12] = HOME
    data.ctrl[:] = data.qpos[:12]; mujoco.mj_forward(model, data)
    with np.load(args.trajectory) as archive:
        actions = archive['actions']
    for action in actions:
        data.ctrl[:] = action
        for _ in range(10):
            mujoco.mj_step(model, data)
            if any(contacts(model, data)):
                details = []
                for c in data.contact:
                    if c.dist >= 0: continue
                    bodies = [model.body(int(model.geom_bodyid[g])).name for g in (c.geom1,c.geom2)]
                    geoms = [model.geom(g).name for g in (c.geom1,c.geom2)]
                    if ('table' in geoms and any(n.startswith(('A_', 'B_')) for n in bodies)) or (any(n.startswith('A_') for n in bodies) and any(n.startswith('B_') for n in bodies)):
                        details.append({'bodies': bodies, 'geoms': geoms, 'penetration_m': -float(c.dist)})
                print(json.dumps({'time_s': data.time, 'contacts': details}, indent=2))
                return
    print('No monitored collision in saved commands.')

if __name__ == '__main__': main()
