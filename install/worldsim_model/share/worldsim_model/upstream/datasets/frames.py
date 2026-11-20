import threading
from collections import OrderedDict
import cv2

import h5video
import numpy as np
import torch
import torch.nn.functional as F


class LRU:

    def __init__(self, read, cap):
        self.read, self.cap, self.items, self.lock = (read, cap, OrderedDict(), threading.Lock())

    def __call__(self, key):
        with self.lock:
            v = self.items.get(key)
            if v is None:
                v = self.items[key] = self.read(*key)
                if len(self.items) > self.cap:
                    self.items.popitem(last=False)
            else:
                self.items.move_to_end(key)
            return v


def bits(data, h, w):
    return np.unpackbits(data)[: h * w].reshape(h, w).astype(bool)


class Frames:

    def __init__(
        self,
        files,
        device,
        gripper=True,
    ):
        """Render and depth targets of packed episodes. Everything comes from the episode file; ``gripper=False`` leaves the
        gripper masks out (minimal recipe)."""
        self.files, self.device, self.gripper = list(files), device, gripper
        self.h5 = {}
        self.render_cache, self.depth_cache = (LRU(self.read_render, 256), LRU(self.read_depth, 16))

    def file(self, fi):
        if fi not in self.h5:
            self.h5[fi] = h5video.File(self.files[fi], "r")
        return self.h5[fi]

    def read_render(self, fi, frame):
        f, cams = (self.file(fi), [])
        for c in [int(x) for x in f.attrs.get("render_cam_ids", f.attrs["cam_ids"])]:
            g = f[f"render/camera_{c}"]
            vm = g["viewmat"][frame].astype(np.float32)
            if not np.isfinite(vm).all():
                continue
            rgb = np.ascontiguousarray(cv2.imdecode(np.asarray(g["jpg"][frame], dtype=np.uint8), cv2.IMREAD_COLOR)[..., ::-1])
            h, w = rgb.shape[:2]
            obj = bits(g["obj_mask"][frame], h, w) if "obj_mask" in g else None
            valid = bits(g["valid_mask"][frame], h, w) if "valid_mask" in g else np.ones((h, w), bool)
            gripper = None
            if self.gripper:
                if "gripper_mask" not in g:
                    raise KeyError(f"{self.files[fi]} has no gripper masks: run scripts/consolidate_packed.py, or train with --recipe minimal")
                gripper = bits(g["gripper_mask"][frame], h, w)
            cams.append(
                dict(
                    rgb=rgb,
                    K=g["K"][frame].astype(np.float32),
                    viewmat=vm,
                    valid=valid,
                    obj=obj,
                    gripper=gripper,
                    H=h,
                    W=w,
                    cid=c,
                )
            )
        return cams

    def warm(self, fi, frame):
        try:
            self.render_cache((fi, frame))
        except Exception:
            pass

    def render(self, fi, frame):
        dev, out = (self.device, [])
        for c in self.render_cache((fi, frame)):
            t = {k: torch.from_numpy(c[k]).to(dev) for k in ("K", "viewmat", "valid")}
            t.update(
                rgb=torch.from_numpy(c["rgb"]).to(dev).float() / 255.0,
                gripper=torch.from_numpy(c["gripper"]).to(dev) if c["gripper"] is not None else None,
                H=c["H"],
                W=c["W"],
                cid=c["cid"],
            )
            t["obj"] = torch.from_numpy(c["obj"]).to(dev) if c["obj"] is not None else None
            out.append(t)
        return out

    def read_depth(self, fi, frame):
        f, cams = (self.file(fi), [])
        for c in [int(x) for x in f.attrs["cam_ids"]]:  # the packed full-scene depth (left-right check applied, 0 = hole)
            try:
                viewmat = np.linalg.inv(f[f"cam/camera_{c}/E_cam2world"][frame].astype(np.float32)).astype(np.float32)
            except np.linalg.LinAlgError:
                continue
            cams.append(
                dict(K=f[f"cam/camera_{c}/K_ir"][frame].astype(np.float32), viewmat=viewmat, cid=c, depth=f[f"img/camera_{c}/depth"][frame])
            )
        return cams

    def depth(self, fi, frame, max_depth=3.0):
        out, dev = ([], self.device)
        for c in self.depth_cache((fi, frame)):
            d = torch.from_numpy(np.ascontiguousarray(c["depth"])).to(dev).float()
            d[~torch.isfinite(d) | (d > max_depth)] = 0.0
            d = d.clamp(min=0)
            dmin = d.clone()
            dmin[dmin <= 0] = float("inf")
            dmin = -F.max_pool2d((-dmin)[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
            out.append(
                dict(
                    K=torch.from_numpy(c["K"]).to(dev),
                    viewmat=torch.from_numpy(c["viewmat"]).to(dev),
                    cid=c["cid"],
                    gt=d,
                    dmin=dmin,
                )
            )
        return out


@torch.no_grad()
def warp_depth(dcams, rcams):
    pws = []
    for dc in dcams:
        gt, K = (dc["gt"], dc["K"])
        vs, us = torch.nonzero(gt > 0, as_tuple=True)
        z = gt[vs, us]
        pc = torch.stack([(us.float() - K[0, 2]) / K[0, 0] * z, (vs.float() - K[1, 2]) / K[1, 1] * z, z], 1)
        E = torch.linalg.inv(dc["viewmat"])
        pws.append(pc @ E[:3, :3].T + E[:3, 3])
    pw = torch.cat(pws)
    outs = []
    for cam in rcams:
        vm, K = (cam["viewmat"], cam["K"])
        pc = pw @ vm[:3, :3].T + vm[:3, 3]
        z = pc[:, 2]
        zs = z.clamp(min=1e-06)
        u = (K[0, 0] * pc[:, 0] / zs + K[0, 2]).long()
        v = (K[1, 1] * pc[:, 1] / zs + K[1, 2]).long()
        ok = (z > 0.05) & (u >= 0) & (u < cam["W"]) & (v >= 0) & (v < cam["H"])
        D = torch.full((cam["H"] * cam["W"],), float("inf"), device=pw.device, dtype=pw.dtype)
        D.scatter_reduce_(0, v[ok] * cam["W"] + u[ok], z[ok], reduce="amin")
        D = D.view(cam["H"], cam["W"])
        D[torch.isinf(D)] = 0.0
        outs.append(D)
    return outs
