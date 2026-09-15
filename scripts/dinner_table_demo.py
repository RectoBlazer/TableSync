"""Static first milestone for the scripted dinner-table workflow."""
import argparse
import json
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from dual_arm_demo import HOME, ROOT, build_scene, write_png


SCENE = ROOT / 'scenes/dinner_table_so101.xml'
OBJECTS = {
    'plate': {'pos': '0 0 0.012', 'size': '0.09 0.065 0.008', 'rgba': '0.92 0.92 0.86 1'},
    'cup': {'pos': '0.19 0 0.045', 'size': '0.025 0.025 0.045', 'rgba': '0.15 0.55 0.85 1'},
    'fork': {'pos': '-0.19 0 0.012', 'size': '0.012 0.055 0.008', 'rgba': '0.85 0.68 0.18 1'},
}


def add_object(world, name, spec):
    body = ET.SubElement(world, 'body', name=name, pos=spec['pos'])
    ET.SubElement(body, 'freejoint', name=f'{name}_free')
    ET.SubElement(body, 'geom', name=f'{name}_geom', type='box', size=spec['size'],
                  mass='0.04', rgba=spec['rgba'], friction='1 0.03 0.003',
                  solref='0.01 1', condim='3', group='3')
    visual = dict(contype='0', conaffinity='0', group='2')
    if name == 'plate':
        ET.SubElement(body, 'geom', type='cylinder', size='0.09 0.006',
                      pos='0 0 0.002', rgba='0.96 0.96 0.91 1', **visual)
        ET.SubElement(body, 'geom', type='cylinder', size='0.073 0.002',
                      pos='0 0 0.009', rgba='0.78 0.84 0.88 1', **visual)
    elif name == 'cup':
        ET.SubElement(body, 'geom', type='cylinder', size='0.03 0.04',
                      rgba='0.12 0.48 0.78 1', **visual)
        ET.SubElement(body, 'geom', type='cylinder', size='0.022 0.0015',
                      pos='0 0 0.0415', rgba='0.035 0.08 0.12 1', **visual)
        ET.SubElement(body, 'geom', type='cylinder', size='0.031 0.002',
                  pos='0 0 0.041', rgba='0.08 0.36 0.62 1', **visual)
    else:
        ET.SubElement(body, 'geom', type='capsule', fromto='0 0.045 0.006 0 -0.035 0.006',
                      size='0.005', rgba='0.92 0.72 0.20 1', **visual)
        ET.SubElement(body, 'geom', type='ellipsoid', size='0.024 0.032 0.006',
                      pos='0 -0.052 0.006', rgba='0.96 0.78 0.26 1', **visual)


def make_model():
    build_scene()
    xml = ET.parse(ROOT / 'scenes/dual_so101.xml')
    world = xml.getroot().find('worldbody')
    for name, spec in OBJECTS.items():
        add_object(world, name, spec)
    ET.indent(xml)
    xml.write(SCENE, encoding='unicode')
    return mujoco.MjModel.from_xml_path(str(SCENE))


def contact_report(model, data):
    arm_table = 0
    arm_arm = 0
    object_table = 0
    for contact in data.contact:
        if contact.dist >= 0:
            continue
        names = [model.body(int(model.geom_bodyid[geom])).name
                 for geom in (contact.geom1, contact.geom2)]
        arms = [name for name in names if name.startswith(('A_', 'B_'))]
        if len(arms) == 2:
            arm_arm += 1
        if 'table' in [model.geom(geom).name for geom in (contact.geom1, contact.geom2)] \
                and arms:
            arm_table += 1
        if any(name in OBJECTS for name in names) \
                and 'table' in [model.geom(geom).name for geom in (contact.geom1, contact.geom2)]:
            object_table += 1
    return {'arm_arm_contacts': arm_arm, 'arm_table_contacts': arm_table,
            'object_table_contacts': object_table}


def initialize(model):
    data = mujoco.MjData(model)
    data.qpos[:12] = np.tile(HOME, 2)
    data.ctrl[:] = data.qpos[:12]
    mujoco.mj_forward(model, data)
    return data


def check(model, data, snapshot=False):
    freejoints = {int(model.jnt_type[model.joint(f'{name}_free').id]) for name in OBJECTS}
    assert freejoints == {mujoco.mjtJoint.mjJNT_FREE}, 'Every object must use a freejoint'
    report = contact_report(model, data)
    assert report['arm_arm_contacts'] == 0, 'Initial arm-to-arm contact detected'
    assert report['arm_table_contacts'] == 0, 'Initial arm-to-table contact detected'
    report.update(object_positions={name: data.body(name).xpos.tolist() for name in OBJECTS},
                  nq=model.nq, nu=model.nu)
    out = ROOT / 'artifacts'
    out.mkdir(exist_ok=True)
    (out / 'dinner_table_check.json').write_text(json.dumps(report, indent=2))
    if snapshot:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, 0.08]
        camera.distance = 1.15
        camera.azimuth = 90
        camera.elevation = -35
        with mujoco.Renderer(model, height=640, width=960) as renderer:
            renderer.update_scene(data, camera=camera)
            write_png(out / 'dinner_table_start.png', renderer.render())
    print(json.dumps(report, indent=2))
    print('PASS: static dinner-table scene loaded with three freejointed objects.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Validate without opening a viewer')
    parser.add_argument('--snapshot', action='store_true', help='Save the initial scene render')
    args = parser.parse_args()
    model = make_model()
    data = initialize(model)
    print('Static milestone: plate center, cup right, fork left; no scripted motion yet.')
    if args.check or args.snapshot:
        check(model, data, snapshot=args.snapshot)
        return
    from mujoco import viewer as mj_viewer
    with mj_viewer.launch_passive(model, data) as view:
        view.cam.lookat[:] = [0, 0, 0.08]
        view.cam.distance = 1.15
        view.cam.azimuth = 90
        view.cam.elevation = -35
        while view.is_running():
            start = time.perf_counter()
            mujoco.mj_step(model, data)
            view.sync()
            time.sleep(max(0, model.opt.timestep - (time.perf_counter() - start)))


if __name__ == '__main__':
    main()