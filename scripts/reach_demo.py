"""Lesson 3: position-only inverse kinematics and sequential shared-point access."""
import argparse
import json
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from dual_arm_demo import ROOT, SCENE, JOINTS, build_scene, write_png

HOME = np.array([-0.7, -1.0, 1.0, 0.0, 0.0, 0.6])
TARGET = np.array([0.0, 0.0, 0.20])  # meters, relative to the table center

def make_model():
    build_scene()
    xml = ET.parse(SCENE)
    world = xml.getroot().find('worldbody')
    world.find("body[@name='B_base']").set('quat', '0 0 0 1')
    ET.SubElement(world, 'site', name='shared_target', pos=' '.join(map(str,TARGET)),
                  type='sphere', size='0.012', rgba='0.1 1 0.2 0.65', group='0')
    for prefix, color in [('A_', '0.1 0.6 1 1'), ('B_', '1 0.3 0.1 1')]:
        site = world.find(f".//site[@name='{prefix}gripperframe']")
        site.set('group','0'); site.set('size','0.007'); site.set('rgba',color)
    path = ROOT/'scenes/reach_so101.xml'
    ET.indent(xml); xml.write(path, encoding='unicode')
    return mujoco.MjModel.from_xml_path(str(path))

def inverse_kinematics(model, prefix, target):
    """Find joint angles on scratch data, without changing the live simulation."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = np.tile(HOME,2)
    joints = [model.joint(prefix+n).id for n in JOINTS[:5]]
    qids = model.jnt_qposadr[joints]
    dofs = model.jnt_dofadr[joints]
    site_id = model.site(prefix+'gripperframe').id
    jac = np.zeros((3,model.nv))
    for _ in range(400):
        mujoco.mj_forward(model,scratch)
        error = target-scratch.site_xpos[site_id]
        if np.linalg.norm(error) < 0.0005:
            return scratch.qpos.copy()
        mujoco.mj_jacSite(model,scratch,jac,None,site_id)
        j = jac[:,dofs]
        step = j.T @ np.linalg.solve(j@j.T + 0.001**2*np.eye(3),error)
        step *= min(1.0,0.08/max(np.max(np.abs(step)),1e-9))
        scratch.qpos[qids] = np.clip(scratch.qpos[qids]+step,
                                     model.jnt_range[joints,0]+0.01,
                                     model.jnt_range[joints,1]-0.01)
    raise RuntimeError(f'{prefix} target not solved; error {np.linalg.norm(error):.4f} m')

def schedule(t, goals):
    """Only one arm leaves home: approach 4 s, hold 2 s, return 4 s, rest 2 s."""
    phase=t%24
    active=0 if phase<12 else 1
    local=phase%12
    if local<4: blend=0.5-0.5*np.cos(np.pi*local/4)
    elif local<6: blend=1.0
    elif local<10: blend=0.5+0.5*np.cos(np.pi*(local-6)/4)
    else: blend=0.0
    home=np.tile(HOME,2)
    return home+blend*(goals[active]-home),active,local

def contacts(model,data):
    cross=table=0
    for c in data.contact:
        if c.dist >= 0: continue
        names=[model.body(int(model.geom_bodyid[g])).name for g in [c.geom1,c.geom2]]
        cross+=int(any(n.startswith('A_') for n in names) and any(n.startswith('B_') for n in names))
        table+=int(any(model.geom(g).name=='table' for g in [c.geom1,c.geom2]) and any(n.startswith(('A_','B_')) for n in names))
    return cross,table

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--snapshot',action='store_true')
    args=parser.parse_args()
    model=make_model()
    goals=[inverse_kinematics(model,p,TARGET) for p in ['A_','B_']]
    data=mujoco.MjData(model)
    data.qpos[:]=np.tile(HOME,2);data.ctrl[:]=data.qpos
    mujoco.mj_forward(model,data)
    camera=mujoco.MjvCamera()
    camera.lookat[:]=[0,0,0.16];camera.distance=1.25;camera.azimuth=90;camera.elevation=-30
    print('Green = shared target. Blue = A gripper reference. Orange = B gripper reference.')
    print('Position-only reach: no object, grasp, orientation constraint or learned policy yet.')
    print('A approaches, holds, retreats; then B. One cycle = 24 simulated seconds.')
    if args.check or args.snapshot:
        errors=[[],[]];cross=table=0
        for _ in range(12000):
            data.ctrl[:],active,local=schedule(data.time,goals)
            mujoco.mj_step(model,data)
            assert np.isfinite(data.qpos).all()
            c,t=contacts(model,data);cross+=c;table+=t
            if 5<local<6:
                pos=data.site(('A_' if active==0 else 'B_')+'gripperframe').xpos
                errors[active].append(float(np.linalg.norm(pos-TARGET)))
            if args.snapshot and abs(data.time-5.5)<0.001:
                out=ROOT/'artifacts';out.mkdir(exist_ok=True)
                with mujoco.Renderer(model,height=640,width=960) as r:
                    r.update_scene(data,camera=camera)
                    write_png(out/'shared_reach.png',r.render())
        report={'target_m':TARGET.tolist(),'hold_max_error_mm':[1000*max(e) for e in errors],
                'arm_contact_steps':cross,'table_contact_steps':table}
        print(json.dumps(report,indent=2))
        assert cross==0 and table==0, 'Trajectory contacts detected'
        assert all(max(e)<0.005 for e in errors), 'Reach error exceeds 5 mm'
        out=ROOT/'artifacts';out.mkdir(exist_ok=True)
        (out/'reach_check.json').write_text(json.dumps(report,indent=2))
        print('PASS: each arm reaches within 5 mm; no inter-arm or arm-table penetration on this trajectory.')
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model,data) as view:
            view.cam.lookat[:]=camera.lookat;view.cam.distance=camera.distance
            view.cam.azimuth=camera.azimuth;view.cam.elevation=camera.elevation
            last=None
            while view.is_running():
                start=time.perf_counter()
                data.ctrl[:],active,local=schedule(data.time,goals)
                label=('A' if active==0 else 'B')+(' approaches' if local<4 else ' holds' if local<6 else ' retreats' if local<10 else ' rests')
                if label!=last: print(label);last=label
                mujoco.mj_step(model,data)
                view.sync()
                time.sleep(max(0,model.opt.timestep-(time.perf_counter()-start)))

if __name__=='__main__':main()
