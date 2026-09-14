"""Lesson 5: physical pick-and-place with measured destination accuracy."""
import argparse
import json
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from grasp_demo import make_model as grasp_model, plan as grasp_plan, pose_ik, run, PICK, BLOCK, HOME
from dual_arm_demo import ROOT, write_png

OFFSET = np.array([0.04, 0.0, 0.0])
DESTINATION = BLOCK[:2] + OFFSET[:2]
PLACEMENT_TOLERANCE = 0.015  # block center must settle within 1.5 cm

def make_model():
    grasp_model()
    xml=ET.parse(ROOT/'scenes/grasp_so101.xml')
    world=xml.getroot().find('worldbody')
    ET.SubElement(world,'site',name='destination',type='box',
                  pos=f'{DESTINATION[0]} {DESTINATION[1]} 0.001',
                  size='0.035 0.035 0.001',rgba='0.1 1 0.2 0.5',group='0')
    path=ROOT/'scenes/place_so101.xml';ET.indent(xml);xml.write(path,encoding='unicode')
    return mujoco.MjModel.from_xml_path(str(path))

def plan(model):
    initial,old=grasp_plan(model)
    d=mujoco.MjData(model);d.qpos[:12]=np.tile(HOME,2);d.qpos[:6]=PICK
    mujoco.mj_forward(model,d)
    pos=d.site('A_gripperframe').xpos.copy()
    rotation=d.site('A_gripperframe').xmat.reshape(3,3).copy()
    place=pose_ik(model,PICK,pos+OFFSET,rotation)
    above=pose_ik(model,place,pos+OFFSET+[0,0,.04],rotation)
    place_closed=place.copy();place_closed[5]=.25
    above_closed=above.copy();above_closed[5]=.25
    return initial,old[:3]+[
        ('carry',4.,above_closed),('hold',3.,above_closed),
        ('lower',3.,place_closed),('release',2.,place),('retreat',3.,above)]

def assess_placement(data):
    error=float(np.linalg.norm(data.body('block').xpos[:2]-DESTINATION))
    assert error <= PLACEMENT_TOLERANCE, f'Placement missed destination by {error*1000:.1f} mm'
    return error

def require_table_support(model,data):
    wanted={model.geom('block_geom').id,model.geom('table').id}
    for i,c in enumerate(data.contact):
        if {c.geom1,c.geom2}==wanted:
            force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
            if force[0]>.01:
                print('  Table support confirmed before release.')
                return
    raise AssertionError('No table support; refusing to release')

def execute(model,data,initial,steps,view=None,snapshot=False):
    report=run(model,data,initial,steps,view=view,snapshot=snapshot,output_stem='place',
               before_release=require_table_support)
    if report is None:return
    report['destination_xy_m']=DESTINATION.tolist()
    report['placement_error_mm']=1000*assess_placement(data)
    report['horizontal_displacement_mm']=1000*float(np.linalg.norm(data.body('block').xpos[:2]-BLOCK[:2]))
    print(f"PASS placement: {report['placement_error_mm']:.1f} mm from destination; "
          f"{report['horizontal_displacement_mm']:.1f} mm from start.")
    out=ROOT/'artifacts';out.mkdir(exist_ok=True)
    (out/'place_check.json').write_text(json.dumps(report,indent=2))
    if snapshot:
        camera=mujoco.MjvCamera();camera.lookat[:]=[-.10,0,.1]
        camera.distance=.85;camera.azimuth=110;camera.elevation=-40
        with mujoco.Renderer(model,height=640,width=960) as renderer:
            renderer.update_scene(data,camera=camera)
            write_png(out/'place_final.png',renderer.render())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--snapshot',action='store_true')
    args=parser.parse_args();model=make_model();initial,steps=plan(model)
    data=mujoco.MjData(model);data.qpos[:12]=np.tile(HOME,2);data.qpos[:6]=initial
    data.ctrl[:]=data.qpos[:12];mujoco.mj_forward(model,data)
    print('Green square = destination, 4 cm toward the table center from the starting point.')
    print('The marker has no collision geometry. The free block is carried by jaw contact.')
    if args.check or args.snapshot:execute(model,data,initial,steps,snapshot=args.snapshot)
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model,data) as view:
            view.cam.lookat[:]=[-.1,0,.1];view.cam.distance=.95;view.cam.azimuth=110;view.cam.elevation=-35
            execute(model,data,initial,steps,view=view)
            while view.is_running():
                mujoco.mj_step(model,data);view.sync();time.sleep(model.opt.timestep)

if __name__=='__main__':main()
