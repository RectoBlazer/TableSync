"""Two arms execute the same physical pick-and-place skill in sequence."""
import argparse
from copy import deepcopy
import json
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from place_demo import make_model as place_model, plan, DESTINATION, PLACEMENT_TOLERANCE
from dual_arm_demo import ROOT, JOINTS, write_png
from reach_demo import contacts

def make_model():
    place_model()
    xml=ET.parse(ROOT/'scenes/place_so101.xml');root=xml.getroot();world=root.find('worldbody')
    # B's base is A's base rotated 180 degrees about the world origin.
    block=deepcopy(world.find("body[@name='block']"))
    pos=np.fromstring(block.get('pos'),sep=' ');pos[:2]*=-1
    block.set('name','block_B');block.set('pos',' '.join(map(str,pos)))
    w,x,y,z=np.fromstring(block.get('quat'),sep=' ')
    block.set('quat',f'{-z} {-y} {x} {w}')
    block.find('freejoint').set('name','block_B_free')
    block.find('geom').set('name','block_B_geom');block.find('geom').set('rgba','1 0.35 0.08 1')
    world.append(block)
    destination=deepcopy(world.find("site[@name='destination']"))
    destination.set('name','destination_B')
    destination.set('pos',f'{-DESTINATION[0]} {-DESTINATION[1]} 0.001')
    world.append(destination)
    grip=root.find("actuator/position[@name='B_gripper']")
    grip.set('kp','20');grip.set('kv','1');grip.set('forcerange','-0.3 0.3')
    path=ROOT/'scenes/bimanual_so101.xml';ET.indent(xml);xml.write(path,encoding='unicode')
    return mujoco.MjModel.from_xml_path(str(path))

def support(model,data,arm):
    """Return surfaces exerting a positive normal force on this arm's block."""
    obj=model.geom('block_geom' if arm=='A' else 'block_B_geom').id
    result=set()
    for i,c in enumerate(data.contact):
        if obj not in (c.geom1,c.geom2):continue
        force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
        if force[0]<=.01:continue
        other=c.geom2 if c.geom1==obj else c.geom1
        if model.geom(other).name=='table':result.add('table')
        name=model.body(int(model.geom_bodyid[other])).name
        if name==arm+'_gripper':result.add('fixed')
        if name==arm+'_moving_jaw_so101_v1':result.add('moving')
    return result

def assess(model,data,arm):
    body='block' if arm=='A' else 'block_B'
    dest=DESTINATION if arm=='A' else -DESTINATION
    pos=data.body(body).xpos.copy()
    error=float(np.linalg.norm(pos[:2]-dest))
    assert error<PLACEMENT_TOLERANCE,f'{arm}: destination missed ({error*1000:.1f} mm)'
    assert abs(pos[2]-.02)<.005,f'{arm}: block not on table'
    assert support(model,data,arm)=={'table'},f'{arm}: release not complete'
    return {'position_m':pos.tolist(),'placement_error_mm':error*1000}

def execute(model,data,initial,steps,view=None,snapshot=False):
    reports={};cross=table=0
    camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,.10]
    camera.distance=1.15;camera.azimuth=90;camera.elevation=-35
    for arm in ['A','B']:
        ids=np.array([model.actuator(arm+'_'+n).id for n in JOINTS])
        last=data.ctrl[ids].copy();heights=[];bilateral=[]
        print(f'WORKSPACE OWNER: {arm}; other arm waits.',flush=True)
        for name,duration,goal in steps+[('park',3.,initial)]:
            if name=='release':
                assert 'table' in support(model,data,arm),f'{arm}: table support missing; release blocked'
            print(f'{arm}: {name}',flush=True)
            n=round(duration/model.opt.timestep)
            for i in range(n):
                wall=time.perf_counter();u=(i+1)/n;s=.5-.5*np.cos(np.pi*u)
                data.ctrl[ids]=last+s*(goal-last)
                mujoco.mj_step(model,data)
                assert np.isfinite(data.qpos).all()
                c,t=contacts(model,data);cross+=c;table+=t
                assert c==0 and t==0,'Robot collision; sequence stopped'
                if name in ['carry','hold']:
                    heights.append(float(data.body('block' if arm=='A' else 'block_B').xpos[2]))
                    bilateral.append({'fixed','moving'}<=support(model,data,arm))
                if view is not None:
                    if not view.is_running():return None
                    view.sync();time.sleep(max(0,model.opt.timestep-(time.perf_counter()-wall)))
            mujoco.mj_forward(model,data)
            if name=='close':
                assert {'fixed','moving'}<=support(model,data,arm),f'{arm}: grasp missing; lift blocked'
            if snapshot and name=='hold':
                out=ROOT/'artifacts';out.mkdir(exist_ok=True)
                with mujoco.Renderer(model,height=640,width=960) as renderer:
                    renderer.update_scene(data,camera=camera)
                    write_png(out/f'bimanual_{arm}_hold.png',renderer.render())
            last=goal.copy()
        assert min(heights)>.05,f'{arm}: lost lift during carry/hold'
        assert sum(bilateral)/len(bilateral)>.9,f'{arm}: grasp contact lost'
        reports[arm]=assess(model,data,arm)
        reports[arm].update(min_carry_hold_height_m=min(heights),
                            bilateral_contact_fraction=sum(bilateral)/len(bilateral))
        print(f'{arm}: placement verified; workspace released.',flush=True)
    # Recheck A after B has acted; completion must hold for both objects at once.
    for arm in ['A','B']:reports[arm].update(assess(model,data,arm))
    reports['inter_arm_contact_steps']=cross;reports['arm_table_contact_steps']=table
    print(json.dumps(reports,indent=2));print('PASS: both blocks placed and released.')
    out=ROOT/'artifacts';out.mkdir(exist_ok=True)
    (out/'bimanual_check.json').write_text(json.dumps(reports,indent=2))
    if snapshot:
        with mujoco.Renderer(model,height=640,width=960) as renderer:
            renderer.update_scene(data,camera=camera)
            write_png(out/'bimanual_final.png',renderer.render())
    return reports

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--snapshot',action='store_true')
    args=parser.parse_args();model=make_model();initial,steps=plan(model)
    data=mujoco.MjData(model);data.qpos[:12]=np.tile(initial,2);data.ctrl[:]=data.qpos[:12]
    mujoco.mj_forward(model,data)
    print('Blue block: Arm A. Orange block: Arm B. Green squares: destinations.')
    print('Sequential scripted skills; fixed object poses; no handoff or learned policy yet.')
    if args.check or args.snapshot:execute(model,data,initial,steps,snapshot=args.snapshot)
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model,data) as view:
            view.cam.lookat[:]=[0,0,.1];view.cam.distance=1.15;view.cam.azimuth=90;view.cam.elevation=-35
            execute(model,data,initial,steps,view=view)
            while view.is_running():
                mujoco.mj_step(model,data);view.sync();time.sleep(model.opt.timestep)

if __name__=='__main__':main()
