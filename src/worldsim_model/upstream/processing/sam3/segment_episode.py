"""Run SAM3 video segmentation on ONE episode/camera (text and/or point
prompts).  segment_all.py is the bulk driver; use this one to debug a prompt
or to hand-fix a camera with clicks (--points-json).

Writes the binary masks to
    <PW_DATA_ROOT>/<dataset>/sam3_masks/episode_<E>_camera_<C>.h5

with the structure
    /mask_union   uint8 (T, H, W)       OR of all object masks per frame
    /mask_per_obj uint8 (T, K, H, W)    per-object binary masks (K = #obj_ids)
    /obj_ids      int32 (K,)            sam3-assigned per-object ids
    /frame_indices int32 (T,)           frame index in source HDF5
    /attrs: dataset, episode, camera, prompt, num_frames, H, W, start_frame, stride

SAM3 is not pip-installed: its checkout ($SAM3_REPO) is added to sys.path.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import h5py

import h5video
import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from processing import paths as ppaths  # noqa: E402

# SAM3 is not pip-installed: env SAM3_REPO (processing.paths).
if str(ppaths.SAM3_REPO) not in sys.path:
    sys.path.insert(0, str(ppaths.SAM3_REPO))


def export_frames_as_jpegs(
    raw_h5: Path, cam: int, frame_indices, out_dir: Path, jpeg_quality: int = 92
):
    """Dump RGB frames for `frame_indices` as 0.jpg, 1.jpg, ... in `out_dir`.

    SAM3's video predictor expects an integer-indexed JPEG folder. We use the
    *position* in `frame_indices` as the sam3 frame index (0..N-1) — the
    original source-HDF5 indices are saved separately in the output H5.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5video.File(raw_h5, "r") as f:
        ds = f[f"observations/images/camera_{cam}_color"]
        H, W = ds.shape[1], ds.shape[2]
        for sam3_idx, src_idx in enumerate(tqdm(frame_indices, desc="dump jpeg")):
            rgb = ds[int(src_idx)]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(out_dir / f"{sam3_idx}.jpg"),
                bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
    return H, W


def propagate_in_video(predictor, session_id, start_frame_index=None):
    outs = {}
    req = dict(type="propagate_in_video", session_id=session_id)
    if start_frame_index is not None:
        # Point-prompt path: the detector's `previous_stages_out` is empty,
        # so without an explicit start frame the inference state raises
        # "No prompts are received on any frames." Forwarding the click
        # frame as the start works around it.
        req["start_frame_index"] = int(start_frame_index)
        req["propagation_direction"] = "forward"
    for response in predictor.handle_stream_request(request=req):
        outs[response["frame_index"]] = response["outputs"]
    return outs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--prompt", type=str, default=None,
                   help="text prompt; mutually exclusive with --points-json")
    p.add_argument("--points-json", type=str, default=None,
                   help="JSON file with point prompts; expected schema: "
                        "{\"<cam_id>\": {\"<obj_id>\": {\"positive\": [[u,v],...], "
                        "\"negative\": [[u,v],...]}, ...}, ...}. "
                        "u,v are in PIXEL coords of the source RGB image.")
    p.add_argument("--num-frames", type=int, default=100,
                   help="how many frames starting from --start-frame")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--stride", type=int, default=1, help="frame subsampling")
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--jpeg-dir", type=str, default=None,
                   help="where to drop the JPEG folder; default $TMPDIR/sam3_<...>")
    p.add_argument("--out", type=str, default=None,
                   help="output HDF5 path; default {sam3_masks_dir}/episode_E_camera_C.h5")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.prompt is None and args.points_json is None:
        raise SystemExit("must pass at least one of --prompt or --points-json "
                         "(passing both runs text first, then refines each "
                         "detected object with the JSON's points).")

    raw_h5 = ppaths.raw_dir(args.dataset) / f"episode_{args.episode}.hdf5"
    if not raw_h5.exists():
        raise FileNotFoundError(raw_h5)

    with h5video.File(raw_h5, "r") as f:
        T_total = f[f"observations/images/camera_{args.camera}_color"].shape[0]
    end = min(T_total, args.start_frame + args.num_frames * args.stride)
    frame_indices = list(range(args.start_frame, end, args.stride))
    n = len(frame_indices)
    print(f"[seg] {args.dataset} ep{args.episode} cam{args.camera} "
          f"frames {args.start_frame}..{frame_indices[-1]} ({n} frames)")

    out_h5 = (Path(args.out) if args.out
              else ppaths.sam3_masks_dir(args.dataset)
              / f"episode_{args.episode}_camera_{args.camera}.h5")
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    if out_h5.exists() and not args.overwrite:
        raise FileExistsError(f"{out_h5} exists; pass --overwrite to clobber")

    # ---- dump JPEGs SAM3 expects an integer-named JPEG folder ----
    jpeg_dir = Path(
        args.jpeg_dir or tempfile.mkdtemp(
            prefix=f"sam3_{args.dataset}_ep{args.episode}_cam{args.camera}_"
        )
    )
    print(f"[seg] dumping {n} jpegs -> {jpeg_dir}")
    H, W = export_frames_as_jpegs(
        raw_h5, args.camera, frame_indices, jpeg_dir,
        jpeg_quality=args.jpeg_quality,
    )

    # ---- build the predictor (downloads from HF cache on first run) ----
    print("[seg] building SAM3 video predictor")
    from sam3.model_builder import build_sam3_video_predictor
    gpus_to_use = [torch.cuda.current_device()] if torch.cuda.is_available() else None
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    # ---- start session + add prompt at frame 0 + propagate ----
    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=str(jpeg_dir))
    )
    session_id = response["session_id"]

    # ---- decide which path: text-only, points-only, or text + point refine ----
    pts_all = None
    cam_pts = None
    if args.points_json is not None:
        with open(args.points_json) as f:
            pts_all = json.load(f)
        cam_key = str(args.camera)
        if cam_key not in pts_all and int(args.camera) in pts_all:
            cam_key = int(args.camera)  # type: ignore
        if cam_key not in pts_all:
            raise SystemExit(
                f"--points-json has no entry for camera {args.camera}; "
                f"keys={list(pts_all.keys())}"
            )
        cam_pts = pts_all[cam_key] or {}

    out0 = None
    if args.prompt is not None:
        prompt_label = f"text={args.prompt!r}"
        print(f"[seg] session_id={session_id}, prompt={args.prompt!r}")
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=args.prompt,
            )
        )
        out0 = response["outputs"]
        text_obj_ids = [int(x) for x in np.asarray(out0["out_obj_ids"]).tolist()]
        print(f"[seg] text detected obj_ids={text_obj_ids}")
        # If user also supplied points: refine each text-detected obj with
        # the JSON points. Two routing conventions are supported:
        #   1. JSON has a key matching one of `text_obj_ids` -> apply to that id
        #   2. Otherwise apply ALL JSON points to the FIRST text-detected id
        #      (typical case: text returns a single obj_id=0 and the picker
        #      JSON was written with obj_id=0 too).
        if cam_pts:
            for obj_id_str, pts in cam_pts.items():
                pos = list(pts.get("positive", []))
                neg = list(pts.get("negative", []))
                if not pos and not neg:
                    continue
                refine_oid = int(obj_id_str)
                if refine_oid not in text_obj_ids:
                    if not text_obj_ids:
                        raise SystemExit(
                            "[seg] text returned 0 objects; cannot refine — "
                            "drop --prompt and use --points-json alone, or "
                            "pick a stronger text prompt."
                        )
                    refine_oid = text_obj_ids[0]
                coords, labels = [], []
                for u, v in pos:
                    coords.append([float(u) / W, float(v) / H]); labels.append(1)
                for u, v in neg:
                    coords.append([float(u) / W, float(v) / H]); labels.append(0)
                print(f"[seg] refining obj_id={refine_oid}  +{len(pos)} -{len(neg)} points")
                response = predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=0,
                        obj_id=refine_oid,
                        points=coords,
                        point_labels=labels,
                    )
                )
                out0 = response["outputs"]
            prompt_label += f" + refine({sum(len(v.get('positive',[]))+len(v.get('negative',[])) for v in cam_pts.values())} pts)"
    else:
        # Point prompts only.
        if not cam_pts:
            raise SystemExit(f"--points-json: no objects for camera {args.camera}")
        prompt_label = f"points={sum(len(v.get('positive',[]))+len(v.get('negative',[])) for v in cam_pts.values())}px across {len(cam_pts)} obj(s)"
        print(f"[seg] session_id={session_id}, {prompt_label}")
        for obj_id_str, pts in cam_pts.items():
            obj_id = int(obj_id_str)
            pos = list(pts.get("positive", []))
            neg = list(pts.get("negative", []))
            if not pos and not neg:
                continue
            coords, labels = [], []
            for u, v in pos:
                coords.append([float(u) / W, float(v) / H]); labels.append(1)
            for u, v in neg:
                coords.append([float(u) / W, float(v) / H]); labels.append(0)
            print(f"  obj_id={obj_id}  +{len(pos)} -{len(neg)} points")
            response = predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    obj_id=obj_id,
                    points=coords,
                    point_labels=labels,
                )
            )
            out0 = response["outputs"]
    obj_ids = np.asarray(out0["out_obj_ids"]).astype(np.int32)
    K = int(obj_ids.size)
    print(f"[seg] frame 0 detected {K} object(s): ids={obj_ids.tolist()}")

    print(f"[seg] propagating across {n} frames…")
    # Text prompts populate the detector path so the inference state can
    # infer where to start. Pure point prompts only populate the tracker
    # path, so we must explicitly pass start_frame_index=0. Text+refine
    # (both) is fine to leave on auto since the text branch fills the
    # detector state.
    start_idx = None if args.prompt is not None else 0
    outputs_per_frame = propagate_in_video(predictor, session_id,
                                           start_frame_index=start_idx)
    # frame 0 may or may not be re-emitted; merge.
    outputs_per_frame.setdefault(0, out0)

    # ---- pack masks into arrays ----
    mask_per_obj = np.zeros((n, K, H, W), dtype=np.uint8)
    mask_union = np.zeros((n, H, W), dtype=np.uint8)

    for fi in range(n):
        out = outputs_per_frame.get(fi)
        if out is None:
            continue
        ids = np.asarray(out["out_obj_ids"]).astype(np.int32).tolist()
        binmasks = out["out_binary_masks"]
        # binmasks: tensor/array (K_fi, H, W)
        if isinstance(binmasks, torch.Tensor):
            binmasks = binmasks.detach().cpu().numpy()
        binmasks = np.asarray(binmasks).astype(bool)
        union = np.zeros((H, W), dtype=bool)
        for k_loc, oid in enumerate(ids):
            if oid not in obj_ids.tolist():
                # propagation may introduce / drop ids; only keep the ones
                # seen at frame 0.
                continue
            k_glob = int(np.where(obj_ids == oid)[0][0])
            m = binmasks[k_loc]
            mask_per_obj[fi, k_glob] = m.astype(np.uint8)
            union |= m
        mask_union[fi] = union.astype(np.uint8)

    nonempty = int((mask_union.sum(axis=(1, 2)) > 0).sum())
    print(f"[seg] frames with any object pixels: {nonempty}/{n} "
          f"(avg fg px / frame = {mask_union.sum() / max(n,1):.0f})")

    # ---- save ----
    print(f"[seg] writing -> {out_h5}")
    with h5py.File(out_h5, "w") as g:
        g.create_dataset("mask_union", data=mask_union,
                         compression="gzip", compression_opts=4)
        g.create_dataset("mask_per_obj", data=mask_per_obj,
                         compression="gzip", compression_opts=4)
        g.create_dataset("obj_ids", data=obj_ids)
        g.create_dataset("frame_indices",
                         data=np.asarray(frame_indices, dtype=np.int32))
        g.attrs["dataset"] = args.dataset
        g.attrs["episode"] = args.episode
        g.attrs["camera"] = args.camera
        g.attrs["prompt"] = args.prompt if args.prompt is not None else prompt_label
        g.attrs["num_frames"] = n
        g.attrs["H"] = H
        g.attrs["W"] = W
        g.attrs["start_frame"] = args.start_frame
        g.attrs["stride"] = args.stride

    print(f"[seg] done. jpeg dir kept at {jpeg_dir}")


if __name__ == "__main__":
    main()
