# External simulation assets

SO-101 model: Google DeepMind MuJoCo Menagerie, `robotstudio_so101`.
Upstream: https://github.com/google-deepmind/mujoco_menagerie
Pinned commit: `8161bba264d7fa7c99ca301e91e7fb44737676ad`.
Model license: Apache 2.0, retained at `mujoco_menagerie/robotstudio_so101/LICENSE`.
Model credits and derivation: upstream model README, retained alongside assets.

To recreate the sparse asset checkout from the project root:

```powershell
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git vendor/mujoco_menagerie
git -C vendor/mujoco_menagerie sparse-checkout set robotstudio_so101
git -C vendor/mujoco_menagerie checkout 8161bba264d7fa7c99ca301e91e7fb44737676ad
```

Our scene builder reuses the original model defaults, masses, meshes and actuators;
it prefixes the robot names and adds a shared table. The lesson trajectory is not
a learned manipulation policy or a validated general collision-avoidance system.

The fourth lesson also adapts the pickup joint pose and block orientation from the
same model's `scene_box.xml`, and documents its changed control/contact parameters
in the project README. Upstream files remain unchanged.
