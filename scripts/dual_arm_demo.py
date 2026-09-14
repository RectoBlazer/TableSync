"""Lesson 2: two real SO-101 models with programmed joint targets, not a policy."""
from pathlib import Path
from copy import deepcopy
import argparse
import time
import xml.etree.ElementTree as ET
import struct
import zlib
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'vendor/mujoco_menagerie/robotstudio_so101/so101.xml'
SCENE = ROOT / 'scenes/dual_so101.xml'
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
HOME = np.array([0.0, -0.5, 0.8, 0.5, 0.0, 0.6])

def build_scene():
    source = ET.parse(SOURCE).getroot()
    scene = ET.Element('mujoco', model='TableSync dual SO101 lesson')
    ET.SubElement(scene, 'compiler', angle='radian', autolimits='true',
                  meshdir='../vendor/mujoco_menagerie/robotstudio_so101/assets')
    ET.SubElement(scene, 'option', timestep='0.002', integrator='implicitfast', gravity='0 0 -9.81')
    visual = ET.SubElement(scene, 'visual')
    ET.SubElement(visual, 'global', offwidth='960', offheight='640')
    scene.append(deepcopy(source.find('default')))
    scene.append(deepcopy(source.find('asset')))
    world = ET.SubElement(scene, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -0.5 2', dir='0 0 -1', diffuse='0.9 0.9 0.9')
    ET.SubElement(world, 'geom', name='table', type='box', pos='0 0 -0.04',
                  size='0.7 0.5 0.04', rgba='0.55 0.60 0.64 1')
    actuators = ET.SubElement(scene, 'actuator')
    for prefix, x in [('A_', -0.32), ('B_', 0.32)]:
        arm = deepcopy(source.find('worldbody/body'))
        arm.set('pos', f'{x} 0 0.003')
        for element in arm.iter():
            if 'name' in element.attrib:
                element.set('name', prefix + element.get('name'))
        world.append(arm)
        for actuator in source.find('actuator'):
            a = deepcopy(actuator)
            a.set('name', prefix + a.get('name'))
            a.set('joint', prefix + a.get('joint'))
            actuators.append(a)
    ET.indent(scene)
    ET.ElementTree(scene).write(SCENE, encoding='unicode')
    return mujoco.MjModel.from_xml_path(str(SCENE))

def targets(t):
    values = np.tile(HOME, 2)
    # A moves first, then B. Smooth pulses start and end at the home target.
    phase = t % 10
    active = 0 if phase < 5 else 6
    pulse = np.sin(np.pi * (phase % 5) / 5) ** 2
    values[active] += 0.25 * pulse
    values[active + 5] += 0.35 * pulse
    return values

def write_png(path, pixels):
    h, w, _ = pixels.shape
    def chunk(kind, payload):
        return struct.pack('!I', len(payload)) + kind + payload + struct.pack('!I', zlib.crc32(kind + payload) & 0xffffffff)
    raw = b''.join(b'\0' + row.tobytes() for row in pixels)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', w,h,8,2,0,0,0)) + chunk(b'IDAT',zlib.compress(raw)) + chunk(b'IEND',b''))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Run 10 seconds without opening a window')
    parser.add_argument('--snapshot', action='store_true', help='Render the final check pose to artifacts')
    args = parser.parse_args()
    model = build_scene()
    data = mujoco.MjData(model)
    assert model.nu == 12 and model.nq == 12
    data.qpos[:] = np.tile(HOME, 2)
    data.ctrl[:] = data.qpos
    mujoco.mj_forward(model, data)
    print('Two SO-101 arms loaded: 12 joints and 12 position actuators.')
    print('Joint order per arm:', ', '.join(JOINTS))
    print('Programmed motion: A turns/opens, returns; then B does the same.')
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.18]
    camera.distance = 1.5
    camera.azimuth = 90
    camera.elevation = -25
    if args.check or args.snapshot:
        cross_contacts = 0
        max_error = 0.0
        for _ in range(5000):
            data.ctrl[:] = targets(data.time)
            mujoco.mj_step(model, data)
            assert np.isfinite(data.qpos).all()
            max_error = max(max_error, float(np.max(np.abs(data.qpos-data.ctrl))))
            for c in data.contact:
                names = [model.body(int(model.geom_bodyid[g])).name for g in [c.geom1,c.geom2]]
                cross_contacts += int(any(n.startswith('A_') for n in names) and any(n.startswith('B_') for n in names))
        assert cross_contacts == 0, f'{cross_contacts} arm-to-arm contacts'
        assert max_error < 0.15, f'Joint tracking error too large: {max_error}'
        print(f'PASS: 10 simulated seconds; no arm-to-arm contacts; max tracking error {max_error:.3f} rad.')
        print('This checks this gentle trajectory only, not general collision avoidance.')
        if args.snapshot:
            out = ROOT / 'artifacts'; out.mkdir(exist_ok=True)
            with mujoco.Renderer(model, height=640, width=960) as renderer:
                renderer.update_scene(data, camera=camera)
                write_png(out/'dual_so101.png', renderer.render())
            print('Saved artifacts/dual_so101.png')
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance = camera.distance
            viewer.cam.azimuth = camera.azimuth
            viewer.cam.elevation = camera.elevation
            while viewer.is_running():
                start = time.perf_counter()
                data.ctrl[:] = targets(data.time)
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(max(0, model.opt.timestep-(time.perf_counter()-start)))

if __name__ == '__main__':
    main()
