"""Physical contact grasp, lift, replace and release. Scripted, not learned."""
import argparse
import json
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from reach_demo import make_model as reach_model, HOME, contacts
from dual_arm_demo import ROOT, JOINTS, write_png

PICK = np.array([0.0, 0.0, 0.473496, 1.17717, 1.58437, 1.0])
BLOCK = np.array([-0.100832, 0.0242023, 0.023])

def make_model():
    reach_model()
    xml=ET.parse(ROOT/'scenes/reach_so101.xml')
    root=xml.getroot();world=root.find('worldbody')
    world.remove(world.find("site[@name='shared_target']"))
    root.find('option').set('iterations','50')
    root.find('option').set('noslip_iterations','5')
    # Limit squeezing torque for the lightweight block; preserve jaw geometry.
    grip=root.find("actuator/position[@name='A_gripper']")
    grip.set('kp','20');grip.set('kv','1');grip.set('forcerange','-0.3 0.3')
    body=ET.SubElement(world,'body',name='block',pos=' '.join(map(str,BLOCK)),
                       quat='0.0643606 -0.704172 0.0643607 0.704172')
    ET.SubElement(body,'freejoint',name='block_free')
    ET.SubElement(body,'geom',name='block_geom',type='box',size='0.02 0.02 0.03',
                  mass='0.04',rgba='0.1 0.55 0.95 1',friction='1 0.03 0.003',solref='0.01 1',condim='3')
    path=ROOT/'scenes/grasp_so101.xml';ET.indent(xml);xml.write(path,encoding='unicode')
    return mujoco.MjModel.from_xml_path(str(path))

def pose_ik(model,seed,target,rotation):
    scratch=mujoco.MjData(model);scratch.qpos[:12]=np.tile(HOME,2)
    scratch.qpos[:5]=seed[:5]
    sid=model.site('A_gripperframe').id
    jp=np.zeros((3,model.nv));jr=jp.copy()
    for _ in range(500):
        mujoco.mj_forward(model,scratch)
        r=scratch.site_xmat[sid].reshape(3,3)
        poserr=target-scratch.site_xpos[sid]
        roterr=sum((np.cross(r[:,i],rotation[:,i]) for i in range(3)))*.5
        if np.linalg.norm(poserr)<.0005 and np.linalg.norm(roterr)<.02:
            return np.r_[scratch.qpos[:5],seed[5]]
        mujoco.mj_jacSite(model,scratch,jp,jr,sid)
        jac=np.vstack([jp[:,:5],.12*jr[:,:5]])
        err=np.r_[poserr,.12*roterr]
        dq=np.linalg.solve(jac.T@jac+1e-5*np.eye(5),jac.T@err)
        dq*=min(1,.08/max(np.max(np.abs(dq)),1e-9))
        scratch.qpos[:5]=np.clip(scratch.qpos[:5]+dq,model.jnt_range[:5,0]+.01,model.jnt_range[:5,1]-.01)
    raise RuntimeError(f'Pose IK failed: position {np.linalg.norm(poserr):.4f} m, rotation {np.linalg.norm(roterr):.4f}')

def plan(model):
    d=mujoco.MjData(model);d.qpos[:12]=np.tile(HOME,2);d.qpos[:6]=PICK
    mujoco.mj_forward(model,d)
    pos=d.site('A_gripperframe').xpos.copy();rot=d.site('A_gripperframe').xmat.reshape(3,3).copy()
    above=pose_ik(model,PICK,pos+[0,0,.04],rot)
    closed=PICK.copy();closed[5]=.25
    raised=above.copy();raised[5]=.25
    # Initial configuration is pregrasp; all subsequent movement uses forces.
    return above,[('approach',3.,PICK),('close',2.,closed),('lift',3.,raised),
                  ('hold',3.,raised),('lower',3.,closed),('release',2.,PICK),('retreat',3.,above)]

def grip_contacts(model,data):
    found=set()
    for index,contact in enumerate(data.contact):
        geoms=[contact.geom1,contact.geom2]
        if model.geom('block_geom').id not in geoms:continue
        force=np.zeros(6);mujoco.mj_contactForce(model,data,index,force)
        if force[0]<=.01:continue
        other=next(g for g in geoms if g!=model.geom('block_geom').id)
        body=model.body(int(model.geom_bodyid[other])).name
        if body=='A_gripper':found.add('fixed')
        elif body=='A_moving_jaw_so101_v1':found.add('moving')
    return found

def run(model,data,initial,steps,view=None,snapshot=False,output_stem='grasp',before_release=None):
    samples=[];contact_steps=[0,0];start_q=initial.copy()
    camera=mujoco.MjvCamera();camera.lookat[:]=[-.05,0,.12]
    camera.distance=1.1;camera.azimuth=120;camera.elevation=-25
    for name,duration,goal in steps:
        if name=='release' and before_release is not None:
            before_release(model,data)
        print(name,flush=True)
        n=round(duration/model.opt.timestep)
        for i in range(n):
            wall=time.perf_counter();u=(i+1)/n;s=.5-.5*np.cos(np.pi*u)
            data.ctrl[:6]=start_q+s*(goal-start_q)
            mujoco.mj_step(model,data)
            assert np.isfinite(data.qpos).all()
            c,t=contacts(model,data);contact_steps[0]+=c;contact_steps[1]+=t
            if name=='hold':samples.append((data.body('block').xpos.copy(),grip_contacts(model,data)))
            if view is not None:
                if not view.is_running():return None
                view.sync();time.sleep(max(0,model.opt.timestep-(time.perf_counter()-wall)))
        mujoco.mj_forward(model,data)
        print('  block z:',round(float(data.body('block').xpos[2]),4),'jaw:',round(float(data.qpos[5]),3),flush=True)
        if name=='close':
            assert grip_contacts(model,data)=={'fixed','moving'}, 'Grasp not confirmed; refusing to lift'
        if snapshot and name=='hold':
            out=ROOT/'artifacts';out.mkdir(exist_ok=True)
            with mujoco.Renderer(model,height=640,width=960) as renderer:
                renderer.update_scene(data,camera=camera)
                write_png(out/('physical_grasp.png' if output_stem=='grasp' else output_stem+'_hold.png'),renderer.render())
        start_q=goal.copy()
    heights=[v[0][2] for v in samples]
    report={'hold_min_height_m':float(min(heights)),
            'hold_height_drift_mm':float(1000*(max(heights)-min(heights))),
            'two_jaw_contact_fraction':sum(v[1]=={'fixed','moving'} for v in samples)/len(samples),
            'final_block_position_m':data.body('block').xpos.tolist(),
            'inter_arm_contact_steps':contact_steps[0],'arm_table_contact_steps':contact_steps[1]}
    print(json.dumps(report,indent=2))
    assert min(heights)>.05,'Block did not stay lifted'
    assert report['two_jaw_contact_fraction']>.9,'No stable two-jaw contact'
    assert report['hold_height_drift_mm']<2.0,'Excessive slip during hold'
    assert contact_steps==[0,0],'Arm collision detected'
    assert abs(data.body('block').xpos[2]-.02)<.005,'Block not replaced on table'
    assert not grip_contacts(model,data),'Block not released'
    print('PASS: physical grasp, sustained lift and release.')
    out=ROOT/'artifacts';out.mkdir(exist_ok=True)
    (out/(output_stem+'_check.json')).write_text(json.dumps(report,indent=2))
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--snapshot',action='store_true')
    args=parser.parse_args();model=make_model();initial,steps=plan(model)
    data=mujoco.MjData(model);data.qpos[:12]=np.tile(HOME,2);data.qpos[:6]=initial
    data.ctrl[:]=data.qpos[:12];mujoco.mj_forward(model,data)
    print('40 g free block. No welds, attachments or runtime object pose changes.')
    if args.check or args.snapshot:run(model,data,initial,steps,snapshot=args.snapshot)
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model,data) as view:
            view.cam.lookat[:]=[-.05,0,.12];view.cam.distance=1.1;view.cam.azimuth=120;view.cam.elevation=-25
            run(model,data,initial,steps,view=view)
            while view.is_running():
                mujoco.mj_step(model,data);view.sync();time.sleep(model.opt.timestep)

if __name__=='__main__':main()
