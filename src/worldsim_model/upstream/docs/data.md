# Data on the Hugging Face Hub

Everything is in one dataset repository, [`jwang771/3d-world-simulator`](https://huggingface.co/datasets/jwang771/3d-world-simulator),
as plain files in the layout the code reads:

```
data/processed/<dataset>/     what training reads: one self-contained file per episode
data/raw/<dataset>/           the recorded episodes; only needed to re-run the processing
checkpoints/                  the released models, the robot arm Gaussians and the scanned room (the interactive demo)
```

## Datasets

| dataset | task | episodes (train / val) | cameras |
|---|---|---|---|
| `cloth` | cloth folding | 210 (200 / 10) | 5 at 848 x 480, colour and stereo infrared |
| `box` | box packing | 112 (102 / 10) | 4 at 1280 x 720, colour and stereo infrared |
| `pusht` | T pushing | 210 (200 / 10) | 4 at 1280 x 720, colour |
| `assembly` | part assembly | 210 (200 / 10) | 4 at 1280 x 720, colour |

Every episode is about 30 s at 10 Hz (297 frames) with 14-d Cartesian actions, joint states and, for `cloth` and
`assembly`, tactile pad readings. The validation episodes are 10 drawn at random per dataset and listed in `splits.json`.
`box` combines three recording sessions (episodes 0-37 continuous teleoperation, 38-97 and 98-111 keyboard teleoperation
at two step sizes); `sources.json` maps every episode to its original session and number.

## Download

```bash
pip install huggingface_hub
hf download jwang771/3d-world-simulator --repo-type dataset --include "data/processed/assembly/*" --local-dir .
python scripts/link_splits.py data/processed/assembly      # train/ and val/ as links into packed/, from splits.json
python scripts/train_vae.py --task assembly --data-root data/processed --out-dir runs/assembly/vae

hf download jwang771/3d-world-simulator --repo-type dataset --include "checkpoints/*" --local-dir .      # the demo
hf download jwang771/3d-world-simulator --repo-type dataset --include "data/raw/assembly/*" --local-dir .   # optional
```

Downloads resume if interrupted. For a first look, one episode is enough:
`--include "data/processed/assembly/packed/episode_0.hdf5" --include "data/processed/assembly/*.json"`.

## Layout of a processed dataset

```
data/processed/<dataset>/
  packed/episode_<E>.hdf5                          one self-contained file per episode: points, depth, colours, masks, render
                                                   targets, robot and gripper labels, tactile targets, actions
  splits.json                                      {"train": [ids], "val": [ids]}; scripts/link_splits.py makes train/ and val/
  sources.json                                     original recording session and episode number of every episode
  config.json, cam_crops.json, table_bbox.json     camera calibration, render crops, table box (used by the processing steps)
```

Training reads `packed/` alone ([data_format.md](data_format.md) lists every field). The folders the processing steps
produce on the way (`fs_disp/` stereo disparity, `sam3_masks/`, `robot_label_cache/`, `gripper_supervision/`,
`tactile_targets/`, `cam_perframe/`) are folded into the episodes by `scripts/consolidate_packed.py` and are not published.

## Storage

Image stacks are stored inside the HDF5 files as lossless video, one FFV1 frame per time step, and read back bit-exactly
through `h5video.File` (a drop-in for `h5py.File`; needs `pip install av`). In the raw recordings this covers the colour
and infrared streams (a third of their original size); in the processed episodes the depth maps. Every converted file
was compared with its source array by array.
