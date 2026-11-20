"""Bulk SAM3 text-prompted segmentation across episodes x cameras.

Loads the SAM3 video predictor ONCE and iterates over every requested
(episode, camera) pair, writing
    <PW_DATA_ROOT>/<ds>/sam3_masks/episode_<E>_camera_<C>.h5
(schema in segment_episode.py).  Needs only the raw colour stream, so it can
run before/independently of FoundationStereo.  ~5 min per episode per GPU
for 5 cams x 300 frames.

Prompts that worked: cloth "green cloth", box "teal box", rope "blue rope"
(rope misses on the top-down cam 0, see processing/README.md).

Usage:
    python -m processing.sam3.segment_all --dataset bimanual_cloth_v3 \\
        --prompt "green cloth" --episodes 0-209 --cams 0,1,2,3,4 --num-frames 400
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[2]))
import h5video  # noqa: E402
from processing import paths  # noqa: E402

# SAM3 is not pip-installed: env SAM3_REPO (processing.paths).
if str(paths.SAM3_REPO) not in sys.path:
    sys.path.insert(0, str(paths.SAM3_REPO))


def parse_episode_range(spec: str) -> list[int]:
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


def export_frames_as_jpegs(raw_h5: Path, cam: int, frame_indices, out_dir: Path,
                           jpeg_quality: int = 92):
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5video.File(raw_h5, "r") as f:
        ds = f[f"observations/images/camera_{cam}_color"]
        H, W = ds.shape[1], ds.shape[2]
        for sam3_idx, src_idx in enumerate(frame_indices):
            rgb = ds[int(src_idx)]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(out_dir / f"{sam3_idx}.jpg"),
                bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
    return H, W


def _union_area(out):
    """Foreground pixel count of an outputs dict's union mask (0 if empty)."""
    bm = out.get("out_binary_masks")
    if bm is None:
        return 0
    if isinstance(bm, torch.Tensor):
        bm = bm.detach().cpu().numpy()
    bm = np.asarray(bm).astype(bool)
    if bm.size == 0 or bm.shape[0] == 0:
        return 0
    return int(bm.any(axis=0).sum())


def _seed_candidates(n, tries):
    """Frame indices to try seeding on: frame 0 first, then evenly spread across
    the clip (so a flaky frame-0 detection isn't the only chance)."""
    if n <= 1:
        return [0]
    spread = np.unique(np.linspace(0, n - 1, max(tries, 1)).round().astype(int)).tolist()
    cands = [0] + [int(c) for c in spread if c != 0]
    return cands[:max(tries, 1)]


def _build_masks(outputs_per_frame, obj_ids, n, H, W):
    """Assemble (mask_per_obj, mask_union, n_kept) numpy arrays from the
    per-frame propagation outputs, keeping only the seed object ids."""
    K = int(obj_ids.size)
    id_list = obj_ids.tolist()
    mask_per_obj = np.zeros((n, max(K, 1), H, W), dtype=np.uint8)
    mask_union = np.zeros((n, H, W), dtype=np.uint8)
    for fi in range(n):
        out = outputs_per_frame.get(fi)
        if out is None:
            continue
        ids = np.asarray(out["out_obj_ids"]).astype(np.int32).tolist()
        binmasks = out["out_binary_masks"]
        if isinstance(binmasks, torch.Tensor):
            binmasks = binmasks.detach().cpu().numpy()
        binmasks = np.asarray(binmasks).astype(bool)
        union = np.zeros((H, W), dtype=bool)
        for k_loc, oid in enumerate(ids):
            if oid not in id_list:
                continue
            k_glob = id_list.index(oid)
            m = binmasks[k_loc]
            mask_per_obj[fi, k_glob] |= m.astype(np.uint8)
            union |= m
        mask_union[fi] = union.astype(np.uint8)
    n_kept = int((mask_union.sum(axis=(1, 2)) > 0).sum())
    return mask_per_obj, mask_union, n_kept


def _propagate_text_fresh_directions(predictor, session_id, seed_frame, prompt, seed):
    """SAM3 may treat a second direction as a cache fetch; reseed it first.

    Reset/reseed makes backward propagation compute its unvisited frames.
    Map refreshed IDs by overlap at the common seed frame, not by ID number.
    """
    def array(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    original_ids = array(seed["out_obj_ids"]).astype(int)
    original_masks = array(seed["out_binary_masks"]).astype(bool)
    outputs = {int(seed_frame): seed}
    for direction in ("forward", "backward"):
        id_map = None
        if direction == "backward":
            if seed_frame == 0:
                continue
            predictor.handle_request(dict(type="reset_session", session_id=session_id))
            refreshed = predictor.handle_request(dict(type="add_prompt", session_id=session_id,
                                      frame_index=int(seed_frame), text=prompt))["outputs"]
            id_map = {}
            for oid, mask in zip(array(refreshed["out_obj_ids"]), array(refreshed["out_binary_masks"]).astype(bool)):
                ious = np.array([(mask & old).sum() / max(1, (mask | old).sum()) for old in original_masks])
                if len(ious) and ious.max() >= .5:
                    id_map[int(oid)] = int(original_ids[ious.argmax()])
        for response in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=session_id,
                                  propagation_direction=direction, start_frame_index=int(seed_frame))):
            fi = int(response["frame_index"])
            out = response["outputs"]
            if id_map is not None:
                out = dict(out)
                out["out_obj_ids"] = np.array([id_map.get(int(x), -1) for x in array(out["out_obj_ids"])])
            if fi != seed_frame:
                outputs[fi] = out
    return outputs


def segment_one(predictor, dataset, episode, camera, prompt, num_frames,
                start_frame, stride, overwrite, quality, out_dir=None,
                seed_tries=20, seed_min_frac=0.005):
    """Run SAM3 on one (episode, camera). Returns (out_path, n_kept, info)."""
    raw_h5 = paths.raw_dir(dataset) / f"episode_{episode}.hdf5"
    if not raw_h5.exists():
        return None, 0, f"missing raw {raw_h5}"
    dest_dir = Path(out_dir) if out_dir is not None else paths.sam3_masks_dir(dataset)
    out_h5 = dest_dir / f"episode_{episode}_camera_{camera}.h5"
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    if out_h5.exists() and not overwrite:
        return out_h5, -1, "skip (exists)"

    with h5video.File(raw_h5, "r") as f:
        T_total = f[f"observations/images/camera_{camera}_color"].shape[0]
    end = min(T_total, start_frame + num_frames * stride)
    frame_indices = list(range(start_frame, end, stride))
    n = len(frame_indices)

    # JPEG dir (cleaned up after the run).
    jpeg_dir = Path(tempfile.mkdtemp(
        prefix=f"sam3_{dataset}_ep{episode}_cam{camera}_"))
    H, W = export_frames_as_jpegs(raw_h5, camera, frame_indices, jpeg_dir,
                                   jpeg_quality=quality)

    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=str(jpeg_dir))
    )
    session_id = response["session_id"]

    seed_frame = 0
    min_seed_area = max(1, int(seed_min_frac * H * W))

    # ---- multi-frame text seeding + propagate-verify-retry ----
    # SAM3's frame-0 text detection is flaky on deformable / low-contrast
    # objects (e.g. cloth): when it whiffs at frame 0 the whole clip ends up
    # empty even though the object is plainly visible. Try the prompt at
    # frames spread across the clip; for each one whose detected area
    # >= min_seed_area, propagate BIDIRECTIONALLY and KEEP it only if the
    # track actually covers the clip (a marginal seed can detect but then
    # collapse to empty). reset_session between tries (clears prompts, keeps
    # loaded frames) so a failed attempt can't poison the next. Stop early
    # once coverage is good; otherwise keep the best of up to `max_prop`
    # propagations.
    max_prop = 4                       # cap full propagations per cam
    accept_cov = 0.60                  # stop early at >=60% frames covered
    best = None                        # (n_kept, seed_frame, mpo, mu)
    n_prop = 0
    for cf in _seed_candidates(n, seed_tries):
        predictor.handle_request(request=dict(type="reset_session",
                                              session_id=session_id))
        o = predictor.handle_request(
            request=dict(type="add_prompt", session_id=session_id,
                         frame_index=int(cf), text=prompt))["outputs"]
        if _union_area(o) < min_seed_area:
            continue                   # no/weak detection at this frame
        oids = np.asarray(o["out_obj_ids"]).astype(np.int32)
        opf = _propagate_text_fresh_directions(predictor, session_id, int(cf), prompt, o)
        mpo, mu, nk = _build_masks(opf, oids, n, H, W)
        n_prop += 1
        if best is None or nk > best[0]:
            best = (nk, int(cf), oids, mpo, mu)
        if nk >= accept_cov * n or n_prop >= max_prop:
            break
    if best is None:                   # nothing detected anywhere
        obj_ids = np.zeros(0, np.int32); K = 0; seed_frame = 0
        mask_per_obj = np.zeros((n, 1, H, W), np.uint8)
        mask_union = np.zeros((n, H, W), np.uint8); n_kept = 0
    else:
        n_kept, seed_frame, obj_ids, mask_per_obj, mask_union = best
        K = int(obj_ids.size)
    with h5py.File(out_h5, "w") as g:
        g.create_dataset("mask_union", data=mask_union,
                         compression="gzip", compression_opts=4)
        g.create_dataset("mask_per_obj", data=mask_per_obj,
                         compression="gzip", compression_opts=4)
        g.create_dataset("obj_ids", data=obj_ids)
        g.create_dataset("frame_indices",
                         data=np.asarray(frame_indices, dtype=np.int32))
        g.attrs["dataset"] = dataset
        g.attrs["episode"] = episode
        g.attrs["camera"] = camera
        g.attrs["prompt"] = f"text={prompt!r}"
        g.attrs["num_frames"] = n
        g.attrs["H"] = H
        g.attrs["W"] = W
        g.attrs["start_frame"] = start_frame
        g.attrs["stride"] = stride
        g.attrs["seed_frame"] = int(seed_frame)

    # Best-effort JPEG dir cleanup.
    try:
        for p in jpeg_dir.glob("*.jpg"):
            p.unlink()
        jpeg_dir.rmdir()
    except Exception:
        pass

    # Free SAM3's per-session state (frame tensors etc.) before next iter
    # or PyTorch leaks ~1GB/episode -> OOM after ~30 eps on a 32 GB GPU.
    try:
        predictor.handle_request(request=dict(type="close_session",
                                              session_id=session_id))
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    avg_fg = int(mask_union.sum() / max(n, 1))
    miss_tag = " MISS(no-detect)" if K == 0 else ""
    return out_h5, n_kept, f"ok n={n} K={K} avg_fg={avg_fg} seed@{seed_frame}{miss_tag}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--prompt", required=True, type=str)
    ap.add_argument("--episodes", required=True,
                    help='episode range, e.g. "0-39" or "0,5,7,12"')
    ap.add_argument("--cams", default="0,1,2,3,4",
                    help='comma-separated cam ids')
    ap.add_argument("--num-frames", type=int, default=1000,
                    help="cap on frames per episode (clipped to the episode length)")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--seed-tries", type=int, default=20,
                    help="try the text prompt at up to this many frames (spread "
                         "across the clip) until detection sticks, then propagate "
                         "bidirectionally from that frame. 1 = old frame-0-only.")
    ap.add_argument("--seed-min-frac", type=float, default=0.005,
                    help="min detected area (fraction of image) to accept a seed "
                         "frame (rejects tiny spurious detections)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="output directory; default <PW_DATA_ROOT>/<ds>/sam3_masks")
    args = ap.parse_args()

    episodes = parse_episode_range(args.episodes)
    cams = [int(c) for c in args.cams.split(",")]
    print(f"[batch] dataset={args.dataset}  prompt={args.prompt!r}")
    print(f"[batch] episodes={episodes}")
    print(f"[batch] cams={cams}  num_frames={args.num_frames}  overwrite={args.overwrite}")

    print("[batch] loading SAM3 video predictor (one-time)…")
    from sam3.model_builder import build_sam3_video_predictor
    gpus = [torch.cuda.current_device()] if torch.cuda.is_available() else None
    t0 = time.time()
    predictor = build_sam3_video_predictor(gpus_to_use=gpus)
    print(f"[batch] model ready in {time.time()-t0:.1f}s")

    tasks = [(e, c) for e in episodes for c in cams]
    print(f"[batch] {len(tasks)} (ep,cam) tasks total")
    t_start = time.time()
    for i, (e, c) in enumerate(tasks):
        t1 = time.time()
        try:
            out, n_kept, info = segment_one(
                predictor, args.dataset, e, c,
                args.prompt, args.num_frames, args.start_frame, args.stride,
                args.overwrite, args.jpeg_quality, out_dir=args.out_dir,
                seed_tries=args.seed_tries, seed_min_frac=args.seed_min_frac,
            )
            dur = time.time() - t1
            eta = (time.time() - t_start) / (i + 1) * (len(tasks) - (i + 1))
            print(f"[batch] [{i+1:3d}/{len(tasks)}] ep{e} cam{c}  "
                  f"{info}  ({dur:.1f}s)  ETA {int(eta//60):02d}:{int(eta%60):02d}",
                  flush=True)
        except Exception as ex:
            print(f"[batch] [{i+1:3d}/{len(tasks)}] ep{e} cam{c}  FAIL: "
                  f"{type(ex).__name__}: {ex}", flush=True)

    print(f"[batch] done in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
