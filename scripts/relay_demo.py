"""Contact-driven table relay of one object. This is not an airborne hand-off."""
import argparse
import json
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from place_demo import make_model as place_model, plan, DESTINATION
from grasp_demo import HOME, BLOCK, pose_ik
from dual_arm_demo import ROOT, write_png
from reach_demo import contacts

# Receiver pickup aligned with the donor's nominal destination.
B_BASE = np.r_[DESTINATION + (BLOCK[:2] - [-.32, 0]), .003]
FINAL = DESTINATION - [.04, 0]

def make_model(seed=None):
    place_model()
    xml = ET.parse(ROOT/'scenes/place_so101.xml')
    root = xml.getroot(); world = root.find('worldbody')
    world.find("body[@name='B_base']").set('pos', ' '.join(map(str, B_BASE)))
    grip = root.find("actuator/position[@name='B_gripper']")
    grip.set('kp', '20'); grip.set('kv', '1'); grip.set('forcerange', '-0.3 0.3')
    ET.SubElement(world, 'site', name='relay_finish', type='box',
                  pos=f'{FINAL[0]} {FINAL[1]} .001', size='.018 .03 .001',
                  rgba='1 .5 .1 .5')
    if seed is not None:
        rng = np.random.default_rng(seed)
        block = world.find("body[@name='block']")
        position = BLOCK.copy(); position[:2] += rng.uniform(-.003, .003, 2)
        block.set('pos', ' '.join(map(str, position)))
        geom = block.find('geom')
        geom.set('mass', str(rng.uniform(.03, .05)))
        geom.set('friction', f'{rng.uniform(.8, 1.2)} .03 .003')
    path = ROOT/'scenes/relay_so101.xml'
    ET.indent(xml); xml.write(path, encoding='unicode')
    return mujoco.MjModel.from_xml_path(str(path))

def support(model, data):
    result = set(); obj = model.geom('block_geom').id
    for i, c in enumerate(data.contact):
        if obj not in (c.geom1, c.geom2): continue
        force = np.zeros(6); mujoco.mj_contactForce(model, data, i, force)
        if force[0] <= .01: continue
        other = c.geom2 if c.geom1 == obj else c.geom1
        if model.geom(other).name == 'table': result.add('table')
        body = model.body(int(model.geom_bodyid[other])).name
        for arm in ('A', 'B'):
            if body == arm+'_gripper': result.add(arm+'_fixed')
            if body == arm+'_moving_jaw_so101_v1': result.add(arm+'_moving')
    return result

def require(condition, message):
    if not condition: raise RuntimeError(message)

def execute(model, data, initial, steps, view=None, snapshot=False, output_dir=None,
            control_hz=None, on_frame=None):
    stride = 1 if control_hz is None else round(1/(control_hz*model.opt.timestep))
    require(stride >= 1, 'Control rate exceeds physics rate')
    if control_hz is not None:
        require(abs(stride*model.opt.timestep-1/control_hz) < 1e-9,
                'Control period must be an integer number of physics steps')
    report = {'transfer_mode': 'table-supported relay', 'arms': {}}
    out = ROOT/'artifacts' if output_dir is None else output_dir
    out.mkdir(parents=True, exist_ok=True)
    camera = mujoco.MjvCamera(); camera.lookat[:] = [-.08, 0, .1]
    camera.distance = .95; camera.azimuth = 90; camera.elevation = -35
    for arm, offset, destination in [('A', 0, DESTINATION), ('B', 6, FINAL)]:
        if arm == 'B':
            require(support(model, data) == {'table'}, 'Transfer blocked: donor has not released onto table')
            require(np.max(np.abs(data.qpos[:6]-HOME)) < .08, 'Transfer blocked: donor has not parked')
            # Simulator feedback corrects the receiver's pickup for placement error.
            # B is rotated by pi, so world XY corrections negate in A's IK frame.
        reference = BLOCK[:2] if arm == 'A' else DESTINATION
        correction = np.r_[(data.body('block').xpos[:2]-reference)*(1 if arm == 'A' else -1), 0.]
        def corrected(goal):
            scratch = mujoco.MjData(model); scratch.qpos[:6] = goal
            mujoco.mj_forward(model, scratch)
            site = scratch.site('A_gripperframe')
            return pose_ik(model, goal, site.xpos.copy()+correction, site.xmat.reshape(3,3).copy())
        # Keep the nominal plan immutable: A's offset must not leak into B's plan.
        local_initial = corrected(initial)
        local_steps = [(name, duration, corrected(goal)) for name, duration, goal in steps]
        print(f'{arm}: measured pickup correction {correction[:2]*1000} mm', flush=True)
        print(f'WORKSPACE OWNER: {arm}', flush=True)
        arm_steps = [('pregrasp', 3., local_initial)] + local_steps + [('park', 4., HOME)]
        heights = []; bilateral = []
        for name, duration, goal in arm_steps:
            if name == 'release': require('table' in support(model, data), 'Release blocked: no table support')
            print(f'{arm}: {name}', flush=True)
            start = data.ctrl[offset:offset+6].copy()
            n = round(duration/model.opt.timestep)
            require(n % stride == 0, 'Stage duration must align with the control period')
            for i in range(n):
                wall = time.perf_counter()
                if i % stride == 0:
                    s = .5-.5*np.cos(np.pi*(i+stride)/n)
                    data.ctrl[offset:offset+6] = start+s*(goal-start)
                    if on_frame is not None:
                        on_frame(model, data, arm, name)
                mujoco.mj_step(model, data)
                require(np.isfinite(data.qpos).all(), 'Non-finite physics state')
                require(contacts(model, data) == (0, 0), f'{arm}/{name}: robot collision')
                if name in ('carry', 'hold'):
                    heights.append(float(data.body('block').xpos[2]))
                    bilateral.append({arm+'_fixed', arm+'_moving'} <= support(model, data))
                if view is not None:
                    if not view.is_running(): return
                    view.sync(); time.sleep(max(0, model.opt.timestep-(time.perf_counter()-wall)))
            if name == 'close':
                require({arm+'_fixed', arm+'_moving'} <= support(model, data), f'{arm}: grasp missing; lift blocked')
            if snapshot and name == 'hold':
                with mujoco.Renderer(model, height=640, width=960) as renderer:
                    renderer.update_scene(data, camera=camera)
                    write_png(out/f'relay_{arm}_hold.png', renderer.render())
        error = float(np.linalg.norm(data.body('block').xpos[:2]-destination))
        require(error < .015, f'{arm}: placement error {error*1000:.1f} mm')
        require(support(model, data) == {'table'}, f'{arm}: object not released')
        require(min(heights) > .05 and np.mean(bilateral) > .9, f'{arm}: carry unstable')
        report['arms'][arm] = {'placement_error_mm': error*1000,
            'pickup_correction_in_ik_frame_mm': (correction[:2]*1000).tolist(),
            'min_carry_height_m': min(heights), 'bilateral_contact_fraction': float(np.mean(bilateral))}
    report['final_position_m'] = data.body('block').xpos.tolist()
    report['inter_arm_and_arm_table_penetrations'] = 0
    (out/'relay_check.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)); print('PASS: same object lifted, carried and released by both arms.')
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true'); parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--seed', type=int, help='Replay a randomized evaluation scene')
    args = parser.parse_args(); model = make_model(args.seed); initial, steps = plan(model)
    data = mujoco.MjData(model); data.qpos[:6] = initial; data.qpos[6:12] = HOME
    data.ctrl[:] = data.qpos[:12]; mujoco.mj_forward(model, data)
    if args.check or args.snapshot: execute(model, data, initial, steps, snapshot=args.snapshot)
    else:
        from mujoco import viewer
        with viewer.launch_passive(model, data) as view:
            view.cam.lookat[:] = [-.08, 0, .1]; view.cam.distance = .95
            view.cam.azimuth = 90; view.cam.elevation = -35
            execute(model, data, initial, steps, view=view)
            while view.is_running():
                mujoco.mj_step(model, data); view.sync(); time.sleep(model.opt.timestep)

if __name__ == '__main__': main()
