"""Small installation checks, not a robot policy or performance benchmark."""
from pathlib import Path
import argparse
import time
import mujoco
import numpy as np
import openvino as ov
from openvino import opset13 as ops

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--view', action='store_true', help='Open an interactive physics viewer')
    args = parser.parse_args()
    print(f'MuJoCo: {mujoco.__version__}')
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'scenes' / 'first_scene.xml'))
    data = mujoco.MjData(model)
    for _ in range(500):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert 0.03 < data.qpos[2] < 0.05, data.qpos
    print(f'PASS physics: cube settled at {data.qpos[2]:.3f} m after {data.time:.1f} s')

    print(f'OpenVINO: {ov.__version__}')
    core = ov.Core()
    print(f'Available OpenVINO devices: {core.available_devices}')
    x = ops.parameter([1, 3], np.float32, name='input')
    network = ov.Model([ops.relu(x)], [x], 'installation_check')
    for device in core.available_devices:
        if device != 'CPU' and not device.startswith('GPU'):
            continue
        compiled = core.compile_model(network, device)
        result = compiled([np.array([[-1, 0, 2]], dtype=np.float32)])[0]
        np.testing.assert_allclose(result, [[0, 0, 2]])
        print(f'PASS inference: {device} / {core.get_property(device, "FULL_DEVICE_NAME")}')

    if args.view:
        from mujoco import viewer as mj_viewer
        mujoco.mj_resetData(model, data)
        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                start = time.perf_counter()
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.perf_counter() - start)))

if __name__ == '__main__':
    main()
