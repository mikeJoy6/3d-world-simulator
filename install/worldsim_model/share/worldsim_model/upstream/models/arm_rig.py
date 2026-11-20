"""The kinematic robot model of the interactive demo: one component, ``ArmRig``, with one call.

The world model predicts the whole scene as Gaussians, arms included, but its arms are soft. The demo therefore draws the arms
from a fitted Gaussian-splat model of the two ViperX 300 S arms (one set of Gaussians per robot link) and keeps the world
model's own Gaussians only at the grippers, where the contact with the scene happens.

    rig = ArmRig("release/arms/rig.pt", "release/cloth/arms.pt", device)
    state = rig.new_state()                             # one per rollout
    arm_gaussians = rig(state, coarse_features, g)      # every frame, in order; g = the world model's Gaussians

Everything the arms need comes from the world model's prediction of that frame, never from the commanded action. The call
has two learned stages and plain kinematics after them:

  where the arms are     a small 3D conv net reads the decoder's coarse features of the latent and gives six joint angles
                         and the finger opening per arm; a one-euro filter with a deadband takes out the frame-to-frame
                         jitter, and an arm whose command did not change this tick holds its joints.
  snap to the gripper    forward kinematics puts the rig at those joints; a PointNet then looks at the world model's robot
                         Gaussians around each rig gripper and gives the small rigid correction that puts the rig gripper
                         exactly on the predicted one. The correction fades out up the forearm, so rig and predicted
                         gripper move as one body.
  hand-over              the rig's Gaussians inside a box around its fingers are hidden; the world model's robot Gaussians
                         are kept only inside that box (``state.boxes``; models/world_model.py applies it).

``release/arms/rig.pt`` holds the rig Gaussians of both arms and the forward-kinematics tables; ``release/<task>/arms.pt``
holds everything that is per task: the arm base frames, both networks and the settings (scripts/export_arms.py writes both).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -- forward kinematics from the exported tables --------------------------------------------------------------------------------
def axis_rotation(axis, q):
    x, y, z = axis
    c, s = np.cos(q), np.sin(q)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s], [y * x * C + z * s, c + y * y * C, y * z * C - x * s], [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def link_poses(fk, q6, opening):
    """Base-frame 4x4 of every rig link (and the gripper link last) at 6 joint angles (rad) and the finger opening (m)."""
    val = np.concatenate([np.asarray(q6, np.float64), [opening, 0.0]])  # index -1: a joint that is not driven
    T = [None] * fk["n_frames"]
    T[fk["base"]] = np.eye(4)
    for par, chi, O, typ, ax, src, mul, off in zip(fk["parent"], fk["child"], fk["origin"], fk["type"], fk["axis"], fk["source"], fk["mul"], fk["off"]):
        x = mul * val[src] + off
        M = O.copy()
        if typ == 1:  # revolute
            M[:3, :3] = O[:3, :3] @ axis_rotation(ax, x)
        elif typ == 2:  # prismatic
            M[:3, 3] = O[:3, 3] + O[:3, :3] @ (ax * x)
        T[chi] = T[par] @ M
    return np.stack([T[i] for i in fk["links"]])


def quat_wxyz(R):
    """(N, 3, 3) rotation matrices -> (N, 4) quaternions."""
    q = np.empty((R.shape[0], 4))
    for n, m in enumerate(R):
        t = np.trace(m)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q[n] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[n] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[n] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[n] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return q


def quat_mul(q, r):
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def partial_transform(T, w, pivot):
    """T^w between the identity and T: the rotation scaled about ``pivot``, the pivot's displacement scaled by w."""
    if w == 1.0:
        return T
    R = T[:3, :3]
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if th < 1e-8:
        Rw = np.eye(3)
    else:
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th)) * th * w
        n = np.linalg.norm(v)
        Rw = np.eye(3) if n < 1e-12 else axis_rotation(v / n, n)
    d = R @ pivot + T[:3, 3] - pivot
    out = np.eye(4)
    out[:3, :3] = Rw
    out[:3, 3] = pivot + w * d - Rw @ pivot
    return out


def rotation_angle_deg(T):
    return float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))


def in_box(x, Wi, lo, hi):
    loc = x @ Wi[:3, :3].T + Wi[:3, 3]
    return ((loc >= lo) & (loc <= hi)).all(1)


# -- the two networks inside ArmRig (not used on their own) -----------------------------------------------------------------------
class _JointNet(nn.Module):
    """Strided 3D conv tower over a decoder feature grid (plus 3 coordinate channels), global mean and max pool, MLP."""

    def __init__(self, channels, width=96, n_out=14):
        super().__init__()
        blk = lambda i, o, s: nn.Sequential(nn.Conv3d(i, o, 3, stride=s, padding=1), nn.GroupNorm(8, o), nn.SiLU())
        self.net = nn.Sequential(blk(channels + 3, width, 2), blk(width, width, 1), blk(width, 2 * width, 2), blk(2 * width, 2 * width, 1), blk(2 * width, 4 * width, 2))
        self.mlp = nn.Sequential(nn.Linear(8 * width, 512), nn.SiLU(), nn.Dropout(0.1), nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, n_out))

    def forward(self, f):
        B, _, D, H, W = f.shape
        zz, yy, xx = torch.meshgrid(torch.linspace(-1, 1, D, device=f.device), torch.linspace(-1, 1, H, device=f.device), torch.linspace(-1, 1, W, device=f.device), indexing="ij")
        h = self.net(torch.cat([f, torch.stack([xx, yy, zz])[None].expand(B, -1, -1, -1, -1)], 1))
        return self.mlp(torch.cat([h.mean((2, 3, 4)), h.amax((2, 3, 4))], 1))


def rot6d(r):
    a, b = r[..., :3], r[..., 3:]
    x = F.normalize(a, dim=-1)
    y = F.normalize(b - (x * b).sum(-1, keepdim=True) * x, dim=-1)
    return torch.stack([x, y, torch.cross(x, y, dim=-1)], -1)


class _LockNet(nn.Module):
    """PointNet over the masked points (xyz in the gripper frame, opacity, colour), mean and max pool, MLP -> rigid transform."""

    def __init__(self, width=128, pool="meanmax", tau=0.2):
        super().__init__()
        self.pool, self.tau = pool, tau
        self.arm_emb = nn.Embedding(2, 16)
        self.pt = nn.Sequential(nn.Linear(7 + 16, width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, 4 * width))
        self.head = nn.Sequential(nn.Linear(8 * width + 16, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 9))

    def forward(self, xyz, op, rgb, mask, arm):
        """xyz (B, N, 3) m, op (B, N), rgb (B, N, 3), mask (B, N) bool, arm (B,) -> (B, 4, 4)."""
        e = self.arm_emb(arm)
        B, N, _ = xyz.shape
        f = self.pt(torch.cat([xyz * 10.0, op[..., None], rgb, e[:, None].expand(B, N, -1)], -1))
        m = mask[..., None].to(f.dtype)
        mean = (f * m).sum(1) / m.sum(1).clamp(min=1)
        if self.pool == "smooth":
            mx = self.tau * torch.logsumexp(f.float().masked_fill(~mask[..., None], -1e4) / self.tau, 1).to(f.dtype)
        else:
            mx = f.masked_fill(~mask[..., None], -1e4).amax(1)
        o = self.head(torch.cat([mean, mx, e], -1)).float()
        T = torch.zeros(B, 4, 4, device=xyz.device)
        T[:, :3, :3] = rot6d(o[:, :6])
        T[:, :3, 3] = o[:, 6:9] / 100.0
        T[:, 3, 3] = 1
        return T


# -- per-session state ------------------------------------------------------------------------------------------------------------
class ArmState:
    """What the arms carry from frame to frame: the joint filter, the last accepted lock correction, and which arms are
    commanded still. After a decode it also holds that frame's results (pose, rig Gaussians, gripper boxes)."""

    def __init__(self):
        self.filter = [None, None]
        self.Tloc = [np.eye(4), np.eye(4)]
        self.still = (False, False)  # set per tick by the caller: this arm's command did not change
        self.head_pose = self.pose = self.rig = self.boxes = None


class ArmRig:

    def __init__(self, rig_path, task_path, device, compile=False):
        R = torch.load(rig_path, map_location="cpu", weights_only=False)
        A = torch.load(task_path, map_location="cpu", weights_only=False)
        self.device, self.cfg = device, A["settings"]
        self.fk = R["fk"]
        self.L = len(self.fk["links"]) - 1
        dev = lambda k: R[k].to(device)
        self.means, self.quats, self.labels = dev("means"), dev("quats"), dev("labels")
        self.static = dict(scale=dev("scale"), opacity=dev("opacity"), rgb01=dev("rgb01"), sh1=dev("sh1"))
        self.template, self.n_template = dev("template"), R["n_template"]  # gripper points the lock crops around
        self.fingers = [x.to(device) for x in R["finger_sub"]]  # finger points the hand-over box is cut around
        self.region, self.region_finger = R["region"], [x.to(device) for x in R["region_finger"]]  # per arm: slice of the wrist + gripper splats
        self.rest_inv = R["rest_inv"].numpy().astype(np.float64)  # (2, L, 4, 4): rig frame of each link at the pose the splats were fitted in
        self.blend = R["attach_w"].numpy().astype(np.float64)  # (L,): share of the lock correction each link takes
        self.base = A["T_base"].numpy().astype(np.float64)  # (2, 4, 4): arm base in the world
        h = A["joints"]
        assert h["kind"] == "conv" and h["feat"] == "coarse", h
        self.joint_net = _JointNet(h["channels"], h["width"]).to(device).eval()
        self.joint_net.load_state_dict(h["state"])
        self.y_mean, self.y_std = h["y_mean"].to(device).float(), h["y_std"].to(device).float()
        k = A["snap"]
        self.lock_net = _LockNet(k["width"], k["pool"], k["tau"]).to(device).eval()
        self.lock_net.load_state_dict(k["state"])
        if compile:
            self.joint_net = torch.compile(self.joint_net, dynamic=False)
        c = self.cfg
        self.lock_cls = torch.tensor(c["lock_cls"], device=device)
        self.back = torch.tensor([[c["grip_back"][0], 0, 0], [c["grip_back"][1], 0, 0]], device=device)
        self.grow = {int(a): torch.tensor(v, device=device) for a, v in c["grip_box_grow"].items()}
        self.arm_ids = torch.tensor([0, 1], device=device)

    def new_state(self):
        return ArmState()

    @torch.no_grad()
    def __call__(self, state, coarse, g):
        """The arms of one frame.
        coarse: the decoder's conv_decoder output for this latent (1, C, D, H, W);
        g: the world model's Gaussians of this frame, before any filter: xyz, opacity, cls and rgb01_lock (unshifted colours).
        Returns the rig's Gaussians dict(xyz, quat, opacity, scale, rgb01, sh1); those handed over to the world model's
        gripper have opacity 0. Afterwards ``state`` holds the frame's pose (14,) = per arm 6 joints (rad) and the finger
        opening (m), and ``state.boxes`` = (Wi (2, 4, 4) world -> gripper frame, lo (2, 3), hi (2, 3)): where the world
        model's robot Gaussians stay."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = self.joint_net(coarse.float())[0].float()
        state.head_pose = (o * self.y_std + self.y_mean).cpu().numpy().astype(np.float64)
        state.pose = self._stabilise(state, state.head_pose)
        return self._posed(state, state.pose, g)

    # -- stage 1: steady joints ------------------------------------------------------------------------------------------------
    def _stabilise(self, state, p, dt=0.1):
        """One-euro filter + soft deadband on each arm's joints (each arm only sees its own speed)."""
        c = self.cfg
        out = p.copy()
        alpha = lambda fc: 1.0 / (1.0 + 1.0 / (2 * np.pi * fc * dt))
        for arm in (0, 1):
            sl = slice(arm * 7, arm * 7 + 7)
            x, st = p[sl], state.filter[arm]
            if st is None or (np.abs(x[:6] - st["x"][:6]).max() > np.radians(20) and st.get("rel", 99) >= 10):  # first frame or a jump: no smoothing
                state.filter[arm] = dict(x=x.copy(), dx=np.zeros(7), shown=x.copy())
                continue
            prev = st["shown"].copy()
            dx = (x - st["x"]) / dt
            st["dx"] = st["dx"] + alpha(1.0) * (dx - st["dx"])
            speed = float(np.linalg.norm(st["dx"][:6]))
            st["x"] = st["x"] + alpha(c["filter_min_cutoff"] + c["filter_beta"] * speed) * (x - st["x"])
            d = st["x"] - st["shown"]
            band = np.r_[np.full(6, np.radians(c["filter_deadband_deg"])), 0.001]
            st["shown"] = st["shown"] + np.sign(d) * np.maximum(np.abs(d) - band, 0.0)
            tol = c["hold_tol_deg"]
            st["far"] = st.get("far", 0) + 1 if (tol > 0 and np.abs(x[:6] - prev[:6]).max() > np.radians(tol)) else 0
            if state.still[arm] and st["far"] < 6:  # commanded still: the joints stay put, fingers and lock still update
                st["shown"][:6] = prev[:6]
                st["x"][:6] = prev[:6]
                st["dx"][:6] = 0.0
                st["rel"] = 0
            elif st.get("rel", 99) < 10:  # just released: glide to the head (at most 3 degrees per joint and frame)
                st["shown"][:6] = prev[:6] + np.clip(st["shown"][:6] - prev[:6], -np.radians(3.0), np.radians(3.0))
                st["rel"] = st.get("rel", 0) + 1
            out[sl] = st["shown"]
        return out

    # -- stage 2: the rig gripper onto the world model's gripper ----------------------------------------------------------
    def _lock_points(self, g):
        c = self.cfg
        m = g["opacity"] > 0.2
        m = m & (g["cls"] == c["lock_cls"][0]) if len(c["lock_cls"]) == 1 else m & torch.isin(g["cls"], self.lock_cls)
        if c["lock_zmin"] is not None:
            m = m & (g["xyz"][:, 2] > c["lock_zmin"])
        return m

    def _snap(self, state, g, S, Wgi):
        """S: the two gripper templates in the world, Wgi (2, 4, 4): world -> uncorrected gripper frame. Sets state.Tloc where accepted."""
        c = self.cfg
        m = self._lock_points(g)
        boxes, inside = [], []
        for arm in (0, 1):
            Wi = Wgi[arm]
            sl = S[arm] @ Wi[:3, :3].T + Wi[:3, 3]
            boxes.append((Wi, sl.min(0).values - 0.025, sl.max(0).values + 0.025))
            inside.append(m & in_box(g["xyz"], *boxes[-1]))
        both = inside[0] & inside[1]
        if c["lock_split"] and bool(both.any()):  # grippers close together: a shared point goes to the nearer template
            P = g["xyz"][both]
            d = [torch.cdist(P, S[arm]).min(1).values for arm in (0, 1)]
            near = (d[0] < 0.01) & (d[1] < 0.01)  # touching grippers: the point stays with both
            inside = [inside[0].clone(), inside[1].clone()]
            inside[0][both] = (d[0] <= d[1]) | near
            inside[1][both] = (d[1] < d[0]) | near
        idx = torch.nonzero(inside[0] | inside[1]).squeeze(1)
        if idx.numel() == 0:
            return
        P, O, C = g["xyz"].index_select(0, idx), g["opacity"].index_select(0, idx), g["rgb01_lock"].index_select(0, idx).clamp(0, 1)
        X = torch.stack([P @ Wi[:3, :3].T + Wi[:3, 3] for Wi, _, _ in boxes])
        M = torch.stack([b.index_select(0, idx) for b in inside])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            T = self.lock_net(X, O[None].expand(2, -1), C[None].expand(2, -1, -1), M, self.arm_ids).float()
        out = torch.cat([T.reshape(-1), M.sum(1).float()]).cpu().numpy().astype(np.float64)  # one download
        T, n = out[:32].reshape(2, 4, 4), out[32:]
        for arm in (0, 1):
            if n[arm] >= 100 and np.linalg.norm(T[arm][:3, 3]) < c["lock_max_t"] and rotation_angle_deg(T[arm]) < c["lock_max_ang"]:
                state.Tloc[arm] = T[arm]  # a correction beyond the limits is rejected: the previous one stays

    # -- kinematics and hand-over ------------------------------------------------------------------------------------------
    def _posed(self, state, pose, g):
        c, dev, L = self.cfg, self.device, self.L
        D, Wg, opening = [], [], []
        for arm in (0, 1):
            opening.append(float(np.clip(pose[arm * 7 + 6], 0.002, 0.12)))
            A = link_poses(self.fk, pose[arm * 7 : arm * 7 + 6], opening[arm])
            D.append(self.base[arm] @ A[:L] @ self.rest_inv[arm])  # rig frame -> world, per link
            Wg.append(self.base[arm] @ A[L])
        up = torch.tensor(np.concatenate([np.concatenate(D).reshape(-1), np.stack([np.linalg.inv(W) for W in Wg]).reshape(-1)]), dtype=torch.float32, device=dev)
        Dt, lt = up[: 2 * L * 16].view(2 * L, 4, 4), self.labels[self.template]
        S = torch.einsum("nij,nj->ni", Dt[lt, :3, :3], self.means[self.template]) + Dt[lt, :3, 3]
        self._snap(state, g, (S[: self.n_template[0]], S[self.n_template[0] :]), up[2 * L * 16 :].view(2, 4, 4))
        Dc, Wgc = [], []
        for arm in (0, 1):
            Tc = Wg[arm] @ state.Tloc[arm] @ np.linalg.inv(Wg[arm])  # the lock correction in the world
            pivot = (Wg[arm] @ np.array([0.09, 0.0, 0.02, 1.0]))[:3]  # blended about the gripper centre
            Dc.append(np.stack([partial_transform(Tc, float(self.blend[l]), pivot) @ D[arm][l] if self.blend[l] > 0 else D[arm][l] for l in range(L)]))
            Wgc.append(Tc @ Wg[arm])
        Dc = np.concatenate(Dc)
        Wi = np.stack([np.linalg.inv(W) for W in Wgc])
        up = torch.tensor(np.concatenate([Dc.reshape(-1), quat_wxyz(Dc[:, :3, :3]).reshape(-1), Wi.reshape(-1)]), dtype=torch.float32, device=dev)  # one upload
        M, qL, Wit = up[: 2 * L * 16].view(2 * L, 4, 4), up[2 * L * 16 : 2 * L * 20].view(2 * L, 4), up[2 * L * 20 :].view(2, 4, 4)
        lab = self.labels
        means = torch.einsum("nij,nj->ni", M[lab, :3, :3], self.means) + M[lab, :3, 3]
        quats = quat_mul(qL[lab], self.quats)
        fl = torch.stack([means[self.fingers[a]] @ Wit[a, :3, :3].T + Wit[a, :3, 3] for a in (0, 1)])  # finger points, gripper frame
        lo = torch.quantile(fl, 0.002, dim=1) - c["grip_pad"] + self.back
        hi = torch.quantile(fl, 0.998, dim=1) + c["grip_pad"]
        opac = self.static["opacity"].clone()
        for a in (0, 1):
            s0, s1 = self.region[a]
            keep = ~in_box(means[s0:s1], Wit[a], lo[a], hi[a])
            if c["rig_overlap"][a] > 0:
                loc = means[s0:s1] @ Wit[a, :3, :3].T + Wit[a, :3, 3]
                keep = keep | ((loc[:, 0] < lo[a, 0] + c["rig_overlap"][a]) & ~self.region_finger[a])
            opac[s0:s1] *= keep.to(opac.dtype)
        # the world model's fingers can be open wider than the joint head says: its box is widened along the finger axis to a
        # set opening, and grown by a fixed amount where an arm carries something rigidly
        lo, hi = lo.clone(), hi.clone()
        for a in (0, 1):
            ex = max(0.0, c["grip_open_box"] - opening[a])
            if ex > 0:
                lo[a, 1] -= ex
                hi[a, 1] += ex
            if a in self.grow:
                lo[a] -= self.grow[a][0::2]
                hi[a] += self.grow[a][1::2]
        state.boxes = (Wit, lo, hi)
        state.rig = dict(xyz=means, quat=quats, opacity=opac, scale=self.static["scale"], rgb01=self.static["rgb01"], sh1=self.static["sh1"])
        return state.rig

    def gripper_anchor(self, state, arm):
        """A point on the gripper body by the wrist camera (world): where the demo draws the arm's gizmo."""
        G = np.linalg.inv(state.boxes[0][arm].cpu().numpy().astype(np.float64))
        return G[:3, :3] @ np.array([0.04, -0.01, 0.09]) + G[:3, 3]
