# Data pipeline (`./pw`)

Raw ALOHA episodes -> FoundationStereo disparity -> SAM3 object masks -> packed
per-episode HDF5 (+ SOR mask, render targets) -> train/ val/ symlinks ->
robot/gripper label sidecars -> preflight.  Everything is driven by
`./pw <cmd> <dataset>`; the python entry points it calls are listed below.

All locations are env-driven (`processing/paths.py`), with repo-relative defaults:

| env | default | meaning |
|---|---|---|
| `ADATA` / `PW_RAW_ROOT` | `raw/` | raw root: `<ds>/episode_N.hdf5`, `<ds>/fs_disp/`, `<ds>/sam3_masks/` |
| `APROC` / `PW_DATA_ROOT` | `data/` | processed root: `<ds>/{raw,fs_disp,sam3_masks}` symlinks + JSONs + packed/ train/ val/ (= `--data-root` of the trainers) |
| `PYTHON` | `python` | interpreter of the `3dws` conda env |
| `FS_REPO` | `external/FoundationStereo` | [FoundationStereo](https://github.com/NVlabs/FoundationStereo) checkout with `pretrained_models/23-51-11` |
| `SAM3_REPO` | `external/sam3` | SAM3 checkout (added to `sys.path`) |
| `TEMPLATE` | `bimanual_box` | dataset whose JSONs seed a new dataset on `./pw setup` |

`pw` exports `PW_DATA_ROOT=$APROC PW_RAW_ROOT=$ADATA`; python scripts resolve
`--dataset <ds>` through `processing.paths`.

Released datasets (`docs/data.md`): `cloth` (5 cams 480x848, wrist cams 3/4 moving), and
`box`, `pusht`, `assembly` (fixed cams 720x1280, wrist cams 3/4 with 360x640 IR).
The names in the examples below (`bimanual_cloth_v3`, ...) are the recording names
they were processed under; `sources.json` of each dataset maps the episodes back.  Wrist cams are listed in `config.json` as `moving_cams` and get
their per-frame pose from `cam_perframe/` sidecars.

## Step by step

```bash
# 0. (optional) IR-only copies for off-site FS: ~40% of raw size, run_fs reads them unchanged
python -m processing.fs_stereo.export_ir_only --src $ADATA/<ds> --dst <dir> --workers 8

# 1. FoundationStereo, BOTH sides (pack's LRC needs disp_right). GPU.
#    0.92 s per 720x1280 frame-side (0.25 s per 360x640 wrist side), 0.39 s per 480x848 side,
#    i.e. ~35 min per box episode, ~2 min per 300-frame cloth_v3 cam.
python -m processing.fs_stereo.run_fs --dataset <ds> --pair both --skip-existing \
    [--src-dir <ir_only_dir> --out-dir $ADATA/<ds>/fs_disp]

# 2. SAM3 masks, every cam, text prompt. GPU, ~5 min per episode (5 cams). Needs only raw colour.
python -m processing.sam3.segment_all --dataset <ds> --prompt "green cloth" \
    --episodes 0-209 --cams 0,1,2,3,4 --num-frames 1000
#    single (ep,cam) debugging / click prompts: processing.sam3.segment_episode

# 3. calibration JSONs (interactive; the user runs these, ports in ./pw help)
./pw setup  <ds>        # $APROC/<ds> + symlinks, JSONs seeded from $TEMPLATE
./pw camera <ds> [ep]   # fixed-cam extrinsics + world bbox   -> config.json
./pw crop   <ds>        # per-cam crop rects (+bbox coverage) -> cam_crops.json, config.json
./pw voxels <ds> [ep]   # sanity: fused cloud + occupancy at the training grid resolution
./pw cams   <ds> [ep]   # sanity: all cams incl. wrist, object projected through the packed poses

# 4. pack -> SOR -> render bake -> split   (LRC=2 was used for every released dataset)
LRC=2 ./pw pack <ds>
./pw tablebbox <ds>     # only if table_bbox.json is missing: pack stops before the render bake,
                        # tune the cam-2 blackout box, then re-run `./pw pack <ds>` (skips done steps)

# 5. robot / gripper label sidecars  (external step, see below)  then the halo bake
python scripts/bake_gripper_halo.py --root $APROC/<ds> --workers 8

# 6. preflight before training
python scripts/preflight_labels.py --root $APROC/<ds>
./pw status <ds>
```

`./pw pack` runs, in order: `pack_dataset --lrc-thresh $LRC --min-depth 0.10
--max-depth 2.0 --workers $WORKERS [--no-obj]` -> bbox/v2 verify ->
`add_sor_mask --sor-k 10 --sor-std 2.0` -> `add_render_to_packed
--blackout-cams $BLACKOUT [--height-mask-cams $HEIGHT_CAMS] --skip-existing` ->
`extract_train_val --val-episodes <id%10==9>` (override with `VAL_EPS`).
Already-packed episodes are skipped unless `OVERWRITE=1`.

### SAM3 prompt notes

* cloth: `"green cloth"` (cloth_v3, all 5 cams).
* box: `"teal box"` (verified on box_v2 cams 0/2).
* thin objects (tried on a rope): `"blue rope"` works on cams 1-4 but MISSES on the top-down cam 0: the
  rope covers less than `--seed-min-frac 0.005` of the image, and lowering the
  threshold tracks badly.  Generic `"rope"` seeds later and propagates backward
  fine but may grab cables as extra objects (check `mask_per_obj`/`obj_ids`).
  `segment_all` tries the prompt at up to `--seed-tries` frames spread across
  the clip and keeps the propagation with the best coverage; `seed_frame` is
  recorded in the attrs.

## Data contract

### raw: `$ADATA/<ds>/episode_N.hdf5`

```
attrs: sim
cartesian_action (T,14) f64        joint_action (T,14) f64
stage (T,) i64                     timestamp (T,) f64
observations/joint_pos (T,14) f32  observations/full_joint_pos (T,16) f32
observations/ee_pos (T,14) f32     observations/robot_base_pose_in_world (T,2,4,4) f32
observations/{left,right}_robot_{left,right}_finger (T,12,32) f32   tactile pads
observations/tactile_receive_timestamp (T,)
observations/images/camera_<c>_color (T,H,W,3) u8
observations/images/camera_<c>_left_ir / _right_ir (T,Hir,Wir) u8     (wrist cams may be 360x640)
observations/images/camera_<c>_intrinsics (T,3,3)   K_ir
observations/images/camera_<c>_rgb_intrinsics (T,3,3)
observations/images/camera_<c>_extrinsics (T,4,4)   world->cam (inverted into E_cam2world)
observations/images/camera_<c>_rgb_to_ir_extrinsic (T,4,4)
observations/images/camera_<c>_baseline (T,) f64
```

### fs_disp: `$ADATA/<ds>/fs_disp/disp_episode_N.hdf5` (run_fs)

```
disp/camera_<c>        (T,Hir,Wir) f16   left-IR-referenced disparity
disp_right/camera_<c>  (T,Hir,Wir) f16   right-IR-referenced (flip trick); REQUIRED by the packer's LRC
meta/camera_<c>/{K_ir (T,3,3), K_rgb (T,3,3), E_cam2world (T,4,4), T_rgb_to_ir (T,4,4), baseline (T,)}
attrs: src_episode, num_frames, valid_iters=32, output_kind='raw_disp', pair/has_right_disp
       (older files: sides='both')
```

### sam3_masks: `$ADATA/<ds>/sam3_masks/episode_N_camera_C.h5`

```
mask_union    (T,H,W) u8      OR of all objects, colour-image frame
mask_per_obj  (T,K,H,W) u8    per object (K = len(obj_ids))
obj_ids       (K,) i32        SAM3 ids; packed label = obj_id + 1
frame_indices (T,) i32        source frame index (stride 1 from 0 in all current datasets)
attrs: dataset, episode, camera, prompt, num_frames, H, W, start_frame, stride, seed_frame
```

### calibration JSONs in `$APROC/<ds>/`

* `config.json` (`processing.config.DatasetConfig`): `cameras.camera_<c>.calibration`
  {translate, rot_local_deg, rot_world_deg} for the FIXED cams, `bbox`
  [xmin,xmax,ymin,ymax,zmin,zmax], `robot_base` {arm: {translate, rot_deg,
  finger_scale, finger_offset_m}}, `tactile_stream_swap`, plus free keys kept
  in `extra`: `moving_cams` ["camera_3","camera_4"],
  `require_moving_camera_sidecars`, `camera_depth_limits`
  {camera: {min_depth, max_depth, max_depth_full}}, `color_calib` {camera:
  {gain, offset}} (applied to cloud colours AND render targets),
  `episode_overrides` [{range:[lo,hi], cameras:{...}, color_calib:{...}}] for
  a camera bumped mid-collection.  Saves archive the previous file in
  `.config_history/`.
* `cam_crops.json`: `{"camera_<c>": [x0, y0, x1, y1]}` colour-image crop per cam
  (render targets).
* `table_bbox.json`: `{"x": [..], "y": [..], "z": [..]}` world box; pixels of the
  blackout cams (default cam 2) whose FS point is outside it are blacked in the
  render target and marked invalid.
* `cam_perframe/ep_<E>_cam<C>.npz`: `E_cam2world (T,4,4)` IR-cam pose per frame
  for each moving cam (plus `E_rgb_cam2world`, `T_base`, `joint_offsets`,
  `T_link_rgb`, `joint_time_offset`, `calibration_kind` for provenance).
  Produced by the external arm-pose fit (below); the packer requires them when
  `require_moving_camera_sidecars` is true, else falls back to the FK pose in
  fs_disp meta.

### packed: `$APROC/<ds>/packed/episode_N.hdf5` (format_version `packed_v2`)

Inspected on `bimanual_cloth_v3/packed/episode_0.hdf5` (T=297, H=480, W=848):

```
attrs
  dataset, episode, src_episode, num_frames, H, W, cam_ids (int32[5]), n_arms=2
  bbox f32[6], robot_base (json {arm: {x,y,z,roll_deg,pitch_deg,yaw_deg,finger_scale,finger_offset_m}})
  occlusion_mask='lrc', lrc_thresh=2.0, min_z=0.1, max_z=2.0, max_z_full=3.5,
  camera_depth_limits (json), depth_semantics='full_scene_lrc',
  mask_semantics='valid=LRC&depth&inframe&bbox; project valid pixels, no load-time filtering'
  tactile_stream_swap, tactile_keys (json list), has_obj_mask, obj_mask_kind='label', obj_labels (json)
  filter_kind='sor', sor_k=10, sor_std=2.0                      (add_sor_mask)
img/camera_<c>/depth       (T,H,W) f16   FULL-SCENE LRC'd depth in IR frame, 0 = hole
img/camera_<c>/rgb_in_ir   (T,) vlen u8  JPEG (q95) of colour sampled into the IR frame at valid px
img/camera_<c>/valid_mask  (T,ceil(HW/8)) u8  bit-packed: LRC & depth & RGB in-frame & bbox
img/camera_<c>/filter_mask (T,ceil(HW/8)) u8  bit-packed: valid & SOR-kept (fused cloud, k=10, std=2)
img/camera_<c>/obj_mask    (T,H,W) u8    label map in IR frame: 0 none, obj_id+1
cam/camera_<c>/K_ir        (T,3,3) f32
cam/camera_<c>/E_cam2world (T,4,4) f32   calibrated (fixed) / sidecar (moving) IR-cam pose
robot/full_joint_pos (T,16) f32   robot/joint_pos (T,14) f32   robot/base_pose_in_world (T,2,4,4) f32
action/joint_action (T,14) f64    action/cartesian_action (T,14) f64
meta/stage (T,) i64               meta/timestamp (T,) f64
tactile/<key> (T,12,32) f32       one per key in tactile_keys
render  attrs: table_bbox (json [x0,x1,y0,y1,z0,z1]), blackout_cams, height_mask_cams [, height_mask_zmax]
render/camera_<c>  attrs: crop [x0,y0,x1,y1], H, W           (add_render_to_packed)
render/camera_<c>/jpg        (T,) vlen u8   cropped colour (q92), colour-calibrated, blackout applied
render/camera_<c>/K          (T,3,3) f32    crop-shifted RGB intrinsics
render/camera_<c>/viewmat    (T,4,4) f32    inv(E_cam2world @ T_rgb_to_ir); NaN = skip frame
render/camera_<c>/obj_mask   (T,ceil(HcWc/8)) u8  bit-packed SAM3 mask_union, cropped
render/camera_<c>/valid_mask (T,ceil(HcWc/8)) u8  blackout / height-mask cams only: supervisable px
```

Point cloud of a frame = lift(depth, K_ir, E_cam2world) at `filter_mask`
(or `valid_mask`) pixels, colour = `rgb_in_ir`, label = `obj_mask`.
Tactile points are URDF-FK'd by the loader from `robot/*`, `tactile/*`,
`robot_base` (the vx300s URDF is not shipped here; `PW_URDF`).

### train/ val/

`extract_train_val` writes relative symlinks `train/episode_N.hdf5 ->
../packed/episode_N.hdf5`; val = every episode with `id % 10 == 9` (cloth_v3:
21 val / 189 train).  Re-running clears and rebuilds both dirs.

### label sidecars (external step, NOT reproducible from this repo)

`robot_label_cache/episode_N.h5` and `gripper_supervision/masks/episode_N.h5`
are produced by the arm-splat fitting tools (fitted vx300s Gaussian-splat
rigs, per-episode base/joint offsets, `models/robot_splats.py`), which are not
part of this repo.  The packer and the VAE loader only consume them:

```
robot_label_cache/episode_N.h5
  attrs: bounds (2,3) [lo; hi] = packed bbox, frames, source
  robot/<l>, gripper/<l>  (T, prod(shape)/8) u8 bit-packed occupancy, attrs shape=(nz,ny,nx)
                          l = 0,1,2 -> 32x16x32, 64x32x64, 128x64x128 (cloth_v3)
gripper_supervision/masks/episode_N.h5
  attrs: episode, frames, source
  camera_<c>/gripper_mask (T, ceil(HcWc/8)) u8, attrs crop/H/W = render crop   (splat silhouette)
gripper_supervision/masks_halo6_surface/episode_N.h5     <- scripts/bake_gripper_halo.py
  same layout, silhouette OR depth surfaces within a 6-fine-voxel halo of gripper/2 (loss ROI only)
```

`cam_perframe/` sidecars for the wrist cams come from the same external fit.
`scripts/preflight_labels.py` fails closed if any episode lacks packed
labels, the render bake, either cache, the halo masks (with crops matching
`cam_crops.json`), or complete SAM3 masks.
