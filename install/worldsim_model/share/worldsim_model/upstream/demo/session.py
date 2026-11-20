"""One visitor's world: the model's state, the keyboard and gizmo control, and the rendered frame.

The control is the dataset's own: every tick a held key (or a gizmo goal not yet reached) moves that arm's commanded position by one
recorded keyboard step along one axis of the arm's base frame; the orientation follows the data's rule (``orient`` in task.json),
positions are clamped to the recorded workspace by default, and a gripper key ramps the gripper command at the recorded rate. The model is
only stepped when something is commanded, so the scene holds still otherwise.

The robot arms are drawn by the kinematic robot model (models/arm_rig.py) when the release has one. Its joints are read from the
world model's latent every frame; the only thing it takes from the control is which arm was commanded still this tick.
"""
import threading
import time
import cv2
import numpy as np
import torch
from models.gsplat_render_loss import render_gaussians
from demo.cells import cells, render_cells

SETTLE_TICKS = 20  # ticks the model keeps running after a gripper toggle, so the object can drop or settle


def aim_rotation(pos, pitch_deg=25.0):
    """Orientation of the cloth and box data: yaw towards the commanded xy, fixed pitch. Returns roll, pitch, yaw."""
    yaw, pitch = np.arctan2(pos[1], pos[0]), np.radians(pitch_deg)
    return np.array([0.0, pitch, yaw])


def to_jpeg(im, quality=80):
    return cv2.imencode(".jpg", np.ascontiguousarray(im[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


class Session:

    def __init__(self, wm, width=960, height=640, fov=50.0, *, clamp_to_workspace=True):
        self.clamp_to_workspace = clamp_to_workspace
        self.wm, self.W, self.H, self.fov = wm, width, height, fov
        cfg, c = wm.cfg, wm.cfg["control"]
        self.keys = {k: tuple(v) for k, v in c["keys"].items()}  # key -> (arm, axis, sign)
        self.grip = c.get("gripper")  # None, or keys {key: arm}, open, close [arm0, arm1], rate, fixed {arm: "open" | "close"}
        self.locked = set(c.get("locked_arms", []))
        self.orient, self.soft_limit, self.run_on_ticks = c["orient"], bool(c.get("soft_limit")), int(c.get("run_on", 0))
        ws = cfg["workspace"]
        self.ws_min, self.ws_max, self.keepouts = np.array(ws["min"], np.float64), np.array(ws["max"], np.float64), ws.get("keepout") or []
        self.base = np.array(cfg["arm_base"], np.float64) if wm.arms is None else wm.arms.base  # (2, 4, 4): each arm's action frame in the world
        self.arms = None  # the kinematic arms' filter state, one per visitor
        v = cfg["view"]
        self.ctl = dict(view="orbit", display="splats", bg=True, arms=True, az=v["az"], el=v["el"], rad=v["radius"], pan=[0.0, 0.0, 0.0], step=c["step"])
        self.z, self.act = wm.load_episode()
        self.lock, self.held, self.taps = threading.Lock(), set(), []
        self.reset_flag, self.redraw, self.stop = True, False, False
        self.jpeg, self.gizmo, self.fps, self.t_seed, self._times = b"", None, 0.0, time.time(), []

    # -- start ---------------------------------------------------------------------------------------------------------------
    def reseed(self):
        self.st = self.wm.seed(self.z, self.act)
        a = self.act[min(self.st.t, len(self.act) - 1)]
        self.S = dict(pos=[a[i * 7 : i * 7 + 3].astype(np.float64) for i in range(2)], rot=[a[i * 7 + 3 : i * 7 + 6].astype(np.float64) for i in range(2)], grip=[float(a[i * 7 + 6]) for i in range(2)])
        if self.grip:
            g = self.grip
            self.S["target"] = ["open" if abs(self.S["grip"][i] - g["open"]) <= abs(self.S["grip"][i] - g["close"][i]) else "close" for i in range(2)]
            for i, t in g.get("fixed", {}).items():
                self.S["target"][int(i)] = t
        self.goal, self.goal_base, self.settle, self.run_on, self.ok_pos = {}, {}, None, 0, {}
        self.arms, self.prev_action = None if self.wm.arms is None else self.wm.arms.new_state(), None
        self._times, self.fps, self.t_seed = [], 0.0, time.time()
        self.z_last = self.st.z_hist[0, -1]
        self.publish()

    # -- gizmo ---------------------------------------------------------------------------------------------------------------
    def gizmo_command(self, d):
        """A drag on the page: {arm, axes, delta (metres along each action axis since the drag started), start, end}."""
        arm = int(d["arm"])
        if arm in self.locked:
            return
        for ax, dv in zip([int(a) for a in d["axes"]], [float(v) for v in d["delta"]]):
            k = (arm, ax)
            if k not in self.goal_base or d.get("start"):
                self.goal_base[k] = float(self.S["pos"][arm][ax])
            target = self.goal_base[k] + dv
            if self.clamp_to_workspace:
                target = np.clip(target, self.ws_min[arm][ax], self.ws_max[arm][ax])
            self.goal[k] = float(target)
            if d.get("end"):
                self.goal_base.pop(k, None)

    def gizmo_driven(self, driven):
        """(arm, axis, sign) for every gizmo goal not yet reached (within half a step); a key on the same axis wins."""
        out = []
        for (arm, ax), goal in list(self.goal.items()):
            if any(d[:2] == (arm, ax) for d in driven):
                continue
            diff = goal - float(self.S["pos"][arm][ax])
            if abs(diff) < 0.5 * self.ctl["step"]:
                if (arm, ax) not in self.goal_base:
                    self.goal.pop((arm, ax))
                continue
            out.append((arm, ax, 1 if diff > 0 else -1))
        return out

    def gizmo_geometry(self, cams, im_w, im_h):
        """Screen geometry for the page: per arm an anchor on the gripper body of the kinematic arm (at the commanded position
        without the arms) and one arrow per keyed axis, in image pixels, with the pixels per metre along each arrow and the map
        from a screen drag of the anchor to the action axes."""
        axes = sorted({(a, x) for a, x, _ in self.keys.values()})
        out = dict(w=im_w, h=im_h, handles=[], anchors=[], goals=[])
        Hm, x0, L = min(c[3] for c in cams), 0.0, 0.06
        for vm, K, W, H in (c[:4] for c in cams):
            V, K, s = vm.cpu().numpy().astype(np.float64), K.cpu().numpy().astype(np.float64), Hm / H

            def project(P):
                q = V[:3, :3] @ P + V[:3, 3]
                return None if q[2] < 0.02 else ((K[0, 0] * q[0] / q[2] + K[0, 2]) * s + x0, (K[1, 1] * q[1] / q[2] + K[1, 2]) * s)

            for arm in sorted({a for a, _ in axes}):
                R = self.base[arm][:3, :3]
                if self.arm_state() is not None and self.arms.boxes is not None:
                    p = self.wm.arms.gripper_anchor(self.arms, arm)
                else:
                    p = R @ self.S["pos"][arm] + self.base[arm][:3, 3]
                a0 = project(p)
                if a0 is None:
                    continue
                depth = float((V[:3, :3] @ p + V[:3, 3])[2])
                m = [[float(R[:, k] @ V[0, :3] * depth / (K[0, 0] * s)), float(R[:, k] @ V[1, :3] * depth / (K[1, 1] * s))] for k in range(3)]
                out["anchors"].append(dict(arm=arm, x=a0[0], y=a0[1], m=m, axes=sorted(x for a, x in axes if a == arm)))
                for aa, ax in axes:
                    a1 = project(p + R[:, ax] * L) if aa == arm else None
                    if a1 is None:
                        continue
                    jx, jy = (a1[0] - a0[0]) / L, (a1[1] - a0[1]) / L
                    if np.hypot(jx, jy) >= 40:  # an axis seen end-on has no usable arrow
                        out["handles"].append(dict(arm=arm, axis=ax, x0=a0[0], y0=a0[1], x1=a1[0], y1=a1[1], ppm=float(np.hypot(jx, jy)), jx=jx, jy=jy))
            x0 += W * Hm / H
        g = self.grip
        out["grips"] = [dict(arm=int(arm), key=k, target=self.S["target"][int(arm)]) for k, arm in sorted(g["keys"].items(), key=lambda kv: kv[1])] if g else []
        return out

    # -- actions -------------------------------------------------------------------------------------------------------------
    def keepout(self, i, pos, lo, hi):
        """The commanded point may not enter this box: a step into it stops at the face it reached and slides along it."""
        lo = np.where(lo <= self.ws_min[i] + 1e-9, -np.inf, lo)
        hi = np.where(hi >= self.ws_max[i] - 1e-9, np.inf, hi)
        inside = lambda p: bool(np.all(p > lo) and np.all(p < hi))
        if not inside(pos):
            return pos
        prev = self.ok_pos.get(i)
        if prev is not None and not inside(prev):
            for k in sorted(range(3), key=lambda k_: abs(pos[k_] - prev[k_])):
                if prev[k] <= lo[k] or prev[k] >= hi[k]:
                    q = np.array(pos, copy=True)
                    q[k] = lo[k] if prev[k] <= lo[k] else hi[k]
                    if not inside(q):
                        return q
            return np.array(prev, copy=True)
        faces = [(abs(f - pos[k]), k, f) for k in range(3) for f in (lo[k], hi[k]) if abs(f) < 9 and self.ws_min[i][k] <= f <= self.ws_max[i][k]]
        if faces:
            _, k, f = min(faces)
            pos = np.array(pos, copy=True)
            pos[k] = f
        return pos

    def build_action(self):
        S, out = self.S, np.zeros(14, np.float32)
        for i in range(2):
            requested = S["pos"][i]
            pos = requested.copy()
            if self.clamp_to_workspace and i not in self.locked:
                pos = np.clip(requested, self.ws_min[i], self.ws_max[i])
                if self.soft_limit:  # a start outside the limits walks back in at key speed
                    pos = requested + np.clip(pos - requested, -self.ctl["step"], self.ctl["step"])
            for kb in self.keepouts:
                if kb["arm"] == i:
                    pos = self.keepout(i, pos, np.asarray(kb["lo"], np.float64), np.asarray(kb["hi"], np.float64))
            self.ok_pos[i] = np.array(pos, copy=True)
            S["pos"][i] = pos
            rot = S["rot"][i] if self.orient == "seed" else aim_rotation(pos)
            out[i * 7 : i * 7 + 7] = np.concatenate([pos, rot, [S["grip"][i]]])
        return out

    def tick(self, held, taps):
        """One control tick: the next 14-d action, or None when nothing is commanded."""
        S, g = self.S, self.grip
        for ch in taps:
            if g and ch in g["keys"]:
                i = g["keys"][ch]
                S["target"][i] = "close" if S["target"][i] == "open" else "open"
        driven = [self.keys[k] for k in held if k in self.keys and self.keys[k][0] not in self.locked]
        driven = driven + self.gizmo_driven(driven)
        ramping = False
        if g:
            want = [g["open"] if S["target"][i] == "open" else g["close"][i] for i in range(2)]
            ramping = any(abs(S["grip"][i] - want[i]) > 1e-9 for i in range(2))
        if driven or ramping:
            prev = [np.array(p, copy=True) for p in S["pos"]]
            for arm, axis, sign in driven:
                S["pos"][arm][axis] += sign * self.ctl["step"]
            if ramping:
                for i in range(2):
                    v, t, r = S["grip"][i], want[i], g["rate"]
                    S["grip"][i] = min(t, v + r) if v < t else max(t, v - r)
            a = self.build_action()
            for arm, ax in list(self.goal):  # a goal the workspace clamp stops short of: drop it
                if any(d[:2] == (arm, ax) for d in driven) and abs(S["pos"][arm][ax] - prev[arm][ax]) < 1e-9 and (arm, ax) not in self.goal_base:
                    self.goal.pop((arm, ax), None)
            if ramping:
                self.settle = 0
            elif all(np.allclose(S["pos"][i], prev[i], atol=1e-9) for i in range(2)) and all(ax == 2 and sg < 0 for _, ax, sg in driven) and self.settle is None:
                return None  # only pushing down into the floor pauses the model
            self.run_on = self.run_on_ticks
            return a
        if self.run_on > 0:  # key released: the real arm lags its command, keep stepping
            self.run_on -= 1
            return self.build_action()
        if self.settle is not None:
            return self.build_action()
        return None

    # -- rendering -----------------------------------------------------------------------------------------------------------
    def arm_state(self):
        """The kinematic arms' state, or None when they are not drawn (no arm files, or the page's raw model output)."""
        return self.arms if self.ctl["arms"] else None

    def cameras(self):
        if self.ctl["view"] == "cams":
            return [c[:4] for c in self.wm.episode_cameras(frame=min(self.st.t, len(self.z) - 1), cams=self.wm.cfg.get("multi_cams"), group=self.wm.cfg.get("multi_view", "render_cams"))]
        return [self.wm.orbit_camera(self.ctl["az"], self.ctl["el"], self.ctl["rad"], W=self.W, H=self.H, fov=self.fov, pan=self.ctl["pan"])]

    def particle_frame(self, z, cam, with_bg):
        """Every Gaussian centre (the model's and the arms') as a solid dot, the room as 1 mm dots on a 4 mm grid, and the
        tactile particles on top as 1.5 mm dots in fire colours (twice the contact signal), on white."""
        wm, dev = self.wm, self.wm.device
        vm, K, W, H = cam
        g = wm.decode(z, arms=self.arm_state())
        if g is not None and self.arm_state() is not None:  # the arm splats handed over to the world model's gripper are hidden, not dots
            shown = ~((g["cls"] == 3) & (g["opacity"] <= 0.003))
            g = {k: v[shown] for k, v in g.items()}
        xyz = g["xyz"] if g is not None else torch.zeros(0, 3, device=dev)
        rgb = g["rgb01"].clamp(0, 1) if g is not None else torch.zeros(0, 3, device=dev)
        size = torch.full((xyz.shape[0], 1), 0.5e-3 * float(wm.cfg.get("point_size_mm", 0.5)), device=dev)
        if with_bg and wm.scan is not None:
            bx, bc = wm.scan.points()
            xyz, rgb, size = torch.cat([xyz, bx]), torch.cat([rgb, bc]), torch.cat([size, torch.full((bx.shape[0], 1), 0.5e-3, device=dev)])
        layers = [(xyz, rgb, size.expand(-1, 3))]
        tac = wm.tactile_particles(z)
        if tac is not None and tac[0].shape[0]:
            q = (tac[1] * 2.0).clamp(0, 1)
            fire = torch.stack([torch.where(q < 0.5, 2 * q, torch.ones_like(q)), torch.where(q < 0.5, torch.zeros_like(q), 2 * (q - 0.5)), torch.zeros_like(q)], -1)
            layers.append((tac[0], fire, torch.full((tac[0].shape[0], 3), 0.75e-3, device=dev)))
        im = torch.ones(H, W, 3, device=dev)
        for x, c, s in layers:
            fg, alpha = render_gaussians(x, s, torch.ones(x.shape[0], device=dev), c, vm[None], K[None], W, H)
            im = fg[0].clamp(0, 1) + (1 - alpha[0]) * im
        gain = float(wm.cfg.get("point_gain", 1.0))
        if gain != 1.0:
            im = (im**2.2 * gain).clamp(0, 1) ** (1 / 2.2)
        return im.mul(255).to(torch.uint8).cpu().numpy()

    def frame(self, z, cams):
        wm, disp, view = self.wm, self.ctl["display"], self.ctl["view"]
        with_bg = bool(self.ctl["bg"]) and view != "orbit_white"
        tiles = []
        if disp in ("latents", "voxels"):
            occ, rgb = cells(wm, z, disp)
            for vm, K, W, H in cams:
                tiles.append(render_cells(occ, rgb, wm.bmin, wm.bmax, vm.cpu().numpy(), K.cpu().numpy(), W, H, srgb=disp == "voxels", gap=0.03 if disp == "latents" else 0.0, ss=2 if disp == "latents" else 1).cpu().numpy())
        elif disp == "particles":
            tiles = [self.particle_frame(z, c, with_bg) for c in cams]
        else:
            g = wm.decode(z, arms=self.arm_state())
            tiles = [wm.render(g, *c, background=with_bg) for c in cams]
        h = min(t.shape[0] for t in tiles)  # all tiles to the same height, side by side
        fit = lambda t: torch.nn.functional.interpolate(torch.from_numpy(t).permute(2, 0, 1)[None].float(), size=(h, int(round(t.shape[1] * h / t.shape[0]))), mode="area")[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()
        return np.concatenate([t if t.shape[0] == h else fit(t) for t in tiles], 1)

    def publish(self):
        cams = self.cameras()
        im = self.frame(self.z_last, cams)
        self.gizmo = self.gizmo_geometry(cams, int(im.shape[1]), int(im.shape[0]))
        self.jpeg = to_jpeg(im)

    # -- page events and the loop --------------------------------------------------------------------------------------------
    def on_keys(self, d):
        with self.lock:
            if "held" in d:
                self.held = set(d["held"])
            if d.get("tap") == "r":
                self.reset_flag = True
            elif "tap" in d:
                self.taps.append(d["tap"])

    def on_control(self, d):
        with self.lock:
            for k in ("view", "display", "bg", "arms", "az", "el", "rad"):
                if k in d:
                    self.ctl[k] = d[k]
            if "dpan" in d:  # shift-drag: move the orbit target in the image plane
                V = self.wm.orbit_camera(self.ctl["az"], self.ctl["el"], self.ctl["rad"])[0].cpu().numpy()
                self.ctl["pan"] = (np.asarray(self.ctl["pan"]) + V[:3, :3].T @ np.array([-d["dpan"][0], -d["dpan"][1], 0.0]) * self.ctl["rad"]).tolist()
            if d.get("reset"):
                self.reset_flag = True
            if "gizmo" in d:
                self.gizmo_command(d["gizmo"])
            self.redraw = True

    def run(self, on_frame, gpu):
        """The session loop (its own thread): ``gpu`` is the lock shared by every session of the process."""
        while not self.stop:
            with gpu:
                if self.reset_flag:
                    self.reset_flag = False
                    self.reseed()
                    on_frame(self.jpeg)
                    continue
                with self.lock:
                    held, taps, self.taps = set(self.held), self.taps, []
                a = self.tick(held, taps)
                if a is not None:
                    if self.arms is not None:  # an arm whose command did not change this tick holds its joints
                        p = self.prev_action
                        self.arms.still = tuple(bool(p is not None and np.allclose(a[i * 7 : i * 7 + 6], p[i * 7 : i * 7 + 6], atol=1e-7)) for i in range(2))
                        self.prev_action = np.array(a, copy=True)
                    self.z_last = self.wm.step(self.st, a)
                    if self.settle is not None:
                        self.settle = None if self.settle + 1 >= SETTLE_TICKS else self.settle + 1
                    self._times = (self._times + [time.time()])[-20:]
                    if len(self._times) > 1:
                        self.fps = (len(self._times) - 1) / (self._times[-1] - self._times[0])
                if a is not None or self.redraw:
                    self.redraw = False
                    self.publish()
                    on_frame(self.jpeg)
            if a is None:
                time.sleep(0.02)
