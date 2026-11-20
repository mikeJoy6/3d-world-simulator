"""A released task folder loaded as one object: the particle-voxel autoencoder's decoder and gaussian head, the latent dynamics
model, and the scanned room. This is the model the interactive demo runs.

    wm = WorldModel("release/cloth")
    z, act = wm.load_episode()                      # the released seed episode (latents + recorded actions)
    state = wm.seed(z, act, frame=24)               # history of n_history latents and actions
    for a in act[state.t:]:                         # any 14-d action: per arm xyz, roll pitch yaw, gripper
        z_next = wm.step(state, a)
        image = wm.render(wm.decode(z_next), *wm.orbit_camera())

With the kinematic robot model of the interactive demo (models/arm_rig.py; needs release/arms and <task>/arms.pt):

    arms = wm.arms.new_state()                      # the arm filter's state: one per rollout, frames decoded in order
    image = wm.render(wm.decode(z_next, arms=arms), *wm.orbit_camera())
"""
import json
from pathlib import Path
import h5py
import numpy as np
import torch

from datasets.latent_seq import read_dense_z
from models.big_head import BigHeadRoundShim
from models.build import load_vae
from models.class_seeds import class_masks, select_class_seeds
from models.dynamics_coarse3d import CoarseDynamics3D
from models.gsplat_render_loss import depth_composite, render_gaussians
from models.scan_background import ScanBackground

TABLE_Z = -0.076


class State:

    def __init__(self, z_hist, a_hist, t):
        self.z_hist, self.a_hist, self.t = (z_hist, a_hist, t)


def orbit_viewmat(target, az_deg, el_deg, radius):
    az, el = (np.deg2rad(az_deg), np.deg2rad(el_deg))
    eye = np.asarray(target, np.float64) + radius * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    f = target - eye
    f /= np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(f, x)
    V = np.eye(4)
    V[:3, :3] = np.stack([x, y, f])
    V[:3, 3] = -V[:3, :3] @ eye
    return V.astype(np.float32)


def _rgb_to_hsv(c):
    r, g, b = c.unbind(-1)
    mx, imx = c.max(-1)
    mn = c.min(-1).values
    d = mx - mn
    dd = d.clamp(min=1e-6)
    h = torch.zeros_like(mx)
    nz = d > 1e-6
    h = torch.where(nz & (imx == 0), ((g - b) / dd) % 6, h)
    h = torch.where(nz & (imx == 1), (b - r) / dd + 2, h)
    h = torch.where(nz & (imx == 2), (r - g) / dd + 4, h)
    return (h * 60.0, torch.where(mx > 1e-6, d / mx.clamp(min=1e-6), torch.zeros_like(mx)), mx)


def _hsv_to_rgb(h, s, v):
    h = (h % 360.0) / 60.0
    i = torch.floor(h)
    f = h - i
    p, q, t = (v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f)))
    i = i.long() % 6
    pick = lambda opts: torch.stack(opts, -1).gather(-1, i[..., None])[..., 0]
    return torch.stack([pick([v, q, p, p, t, v]), pick([t, v, v, q, p, p]), pick([p, p, t, v, v, q])], -1)


def gripper_hue_shift(rgb01, cls, deg=8.0):
    """The red gripper pads read orange on the real robot: shift the hue of red robot-class gaussians by up to ``deg`` degrees."""
    h, s, v = _rgb_to_hsv(rgb01.clamp(0, 1))
    hh = torch.where(h > 180, h - 360, h)
    w = ((40 - hh) / 20).clamp(0, 1) * (hh > -25).float() * ((s - 0.30) / 0.15).clamp(0, 1) * (v > 0.15).float() * (cls == 1).float()
    sel = w > 0
    if not bool(sel.any()):
        return rgb01
    out = rgb01.clone()
    out[sel] = _hsv_to_rgb(hh[sel] + deg * w[sel], s[sel], v[sel])
    return out


class WorldModel:

    def __init__(self, task_dir, device="cuda", bf16=False, background=True, fast=False, arms=True):
        """fast: the interactive settings (bf16, the finest decoder level evaluated only near the surface, compiled dynamics);
        renders differ from the reference by well under 1/255.
        arms: load the kinematic robot model when the task folder has one (``self.arms``, else None); see ``decode``."""
        bf16 = bf16 or fast
        self.dir, self.device, self.bf16 = (Path(task_dir), device, bf16)
        self.cfg = json.load(open(self.dir / "task.json"))
        ck = torch.load(self.dir / "vae.pth", map_location="cpu", weights_only=False)
        self.vae = load_vae(ck, device)
        self.features = {}
        self.vae.decoder.up_finest.register_forward_hook(lambda m, i, o: self.features.__setitem__("feat_2", o))
        spec = ck["new_head"]
        self.head = BigHeadRoundShim(feat_dim=64, K=spec["children"], shape=spec["shape"], cap=spec["cap"]).to(device).eval()
        self.head.load_state_dict(ck["seed_head_state_dict"])
        self.head.head.bf16 = True  # the Gaussian head always runs in bfloat16, as in the public demo (frames then match it to 60 dB or better)
        self.K = spec["children"]
        self.bmin, self.bmax = (self.vae.bounds_min_world, self.vae.bounds_max_world)
        self.vsize = (self.bmax - self.bmin) / torch.tensor(self.vae.grid_shapes[-1], device=device, dtype=torch.float32)
        dk = torch.load(self.dir / "dynamics.pth", map_location="cpu", weights_only=False)
        a = dk["args"]
        self.n_hist = int(a["n_history"])
        self.hist_noise = float(a.get("infer_hist_noise", 0.1))
        self.dyn = CoarseDynamics3D(
            latent_channels=8, action_dim=14, n_history=self.n_hist, n_future=1, width=a.get("dim", 56), action_emb_dim=128,
            n_levels=a.get("n_levels", 3), film=a.get("film", True),
        ).to(device).eval()
        self.dyn.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in dk["model_state_dict"].items()})
        self.amean = np.asarray(dk["action_mean"], np.float32)
        self.astd = np.maximum(np.asarray(dk["action_std"], np.float32), 1e-9)
        self.scan = None
        bg = self.cfg.get("background", "../background/scan_room.pt")
        if background and bg:
            sc = ck.get("scan_color")
            self.scan = ScanBackground(
                self.dir / bg, self.bmin, self.bmax, device, color_gain=None if sc is None else sc["gain"][0],
                color_bias=None if sc is None else sc["bias"][0], white_balance=self.cfg.get("scan_white_balance"),
                room_gain=self.cfg.get("scan_room_gain", 1.0), wall_saturation=self.cfg.get("scan_wall_saturation"), table_gain=self.cfg.get("scan_table_gain"),
            )
        self.sparse = None
        if fast:
            from models.sparse_decoder import SparseDecoder

            self.sparse = SparseDecoder(self.vae.decoder)
            self.dyn = torch.compile(self.dyn)
        self.exposure = float(self.cfg.get("exposure", 1.0))
        self.class_filters = bool(self.cfg.get("class_filters", True))
        self.table_feather = float(self.cfg.get("table_feather", 0.15))
        self.table_edge_faces = self.cfg.get("table_edge_faces")  # hard cut of the table layer per bbox face: x low, x high, y low, y high (m)
        self.arms = None
        rig = self.dir / self.cfg.get("arms", "../arms/rig.pt")
        if arms and rig.is_file() and (self.dir / "arms.pt").is_file():
            from models.arm_rig import ArmRig

            self.arms = ArmRig(rig, self.dir / "arms.pt", device, compile=fast)

    # -- episodes and actions ----------------------------------------------------------------------------------------------
    def load_episode(self, path=None):
        """latents (T, C, nz, ny, nx) float32 and recorded actions (T, 14) of a latent episode file (default: the task's seed)."""
        path = path or self.dir / self.cfg["seed"]["file"]
        with h5py.File(path, "r") as f:
            return (read_dense_z(f), f["action"][:].astype(np.float32))

    def norm_action(self, a):
        return ((np.asarray(a, np.float32) - self.amean) / self.astd).astype(np.float32)

    def seed(self, z, act, frame=None):
        """Start from ``frame`` of an episode: the model's history is the next n_history latents and actions."""
        F0 = self.cfg["seed"]["frame"] if frame is None else int(frame)
        F0 = min(max(F0, 0), len(z) - self.n_hist - 1)
        z_hist = torch.from_numpy(z[F0 : F0 + self.n_hist]).to(self.device)[None]
        a_hist = torch.from_numpy(self.norm_action(act[F0 : F0 + self.n_hist])).to(self.device)[None]
        return State(z_hist, a_hist, F0 + self.n_hist)

    # -- one dynamics step ---------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, state, action):
        """Predict the next latent for a raw 14-d action, advance the state, and return the latent (C, nz, ny, nx)."""
        a_next = torch.from_numpy(self.norm_action(action)).to(self.device)[None, None]
        z_in = torch.cat([state.z_hist, torch.randn_like(state.z_hist[:, :1])], 1)
        noise = torch.tensor([[self.hist_noise] * self.n_hist + [1.0]], device=self.device)
        a_all = torch.cat([state.a_hist, a_next], 1).permute(1, 0, 2)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            z = self.dyn(z_in, a_all, noise_std=noise, future_only=True)[:, -1:].float()
        state.z_hist = torch.cat([state.z_hist[:, 1:], z], 1)
        state.a_hist = torch.cat([state.a_hist[:, 1:], a_next], 1)
        state.t += 1
        return z[0, 0]

    # -- latent to gaussians -------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def decode_cells(self, z):
        """Decoder outputs of one latent: occupancy logits per level, the finest features the Gaussian head reads, and the
        coarse features (``coarse``) the kinematic arm model reads."""
        z = torch.as_tensor(z, device=self.device, dtype=torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            coarse = self.vae.conv_decoder(z[None])
            if self.sparse is not None:
                return dict(self.sparse(coarse), coarse=coarse)
            out = self.vae.decoder(coarse, occ_mask_0=None, use_gt_mask=False)
        return dict(out, feat_2=self.features["feat_2"], coarse=coarse)

    def _isolated(self, xyz, m, cell=0.02, frac=0.03, iters=40):
        """Members of mask ``m`` in a small detached piece: connected components of the occupied 2 cm cells; pieces with fewer
        than ``frac`` of the largest piece's gaussians are flagged (stray object-class blobs near the wrist cameras)."""
        lo = self.bmin - cell
        dims = [int(np.ceil(float(h - l) / cell)) + 3 for l, h in zip(self.bmin, self.bmax)]
        n = dims[0] * dims[1] * dims[2]
        ijk = torch.minimum(((xyz - lo) / cell).long().clamp_min(0), torch.tensor(dims, device=xyz.device) - 1)
        flat = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
        cnt = torch.zeros(n, device=xyz.device).index_add_(0, flat, m.float())
        occ = (cnt > 0).float()
        lab = (torch.arange(1, n + 1, device=xyz.device, dtype=torch.float32) * occ).view(1, 1, *dims)
        occ = occ.view(1, 1, *dims)
        for _ in range(iters):
            lab = torch.nn.functional.max_pool3d(lab, 3, 1, 1) * occ
        lab = lab.view(-1).long()
        size = torch.zeros(n + 1, device=xyz.device).index_add_(0, lab, cnt)
        size[0] = 0
        return m & (size[lab[flat]] < frac * size.max())

    @torch.no_grad()
    def decode_raw(self, z):
        """Every Gaussian the head predicts for one latent, before the display filters, and the decoder's coarse features."""
        z = torch.as_tensor(z, device=self.device, dtype=torch.float32)
        out = self.decode_cells(z)
        masks = class_masks(out["scene_occ_2"].float(), out["tac_occ_2"].float(), out["signal_2"].float(), cloth_dilate=1)
        seeds, cls = select_class_seeds(masks)
        if seeds.shape[0] == 0:
            return None, out["coarse"]
        _, r = self.head(out["feat_2"].float(), seeds, self.bmin, self.vsize)
        g = dict(
            xyz=r["xyz"].reshape(-1, 3), scale=r["scale"].reshape(-1, 3), opacity=r["opacity"].reshape(-1),
            rgb01=((r["rgb"].clamp(-1, 1) + 1) / 2).reshape(-1, 3), quat=r["quat"].reshape(-1, 4), cls=cls.repeat_interleave(self.K),
        )
        g["rgb01_lock"] = g["rgb01"]  # the kinematic arm model was trained on the unshifted colours
        g["rgb01"] = gripper_hue_shift(g["rgb01"], g["cls"])
        return g, out["coarse"]

    @torch.no_grad()
    def decode(self, z, arms=None):
        """Gaussians of one latent: dict(xyz, scale, opacity, rgb01, quat, cls) with cls 0 object, 1 robot, 2 table; None if empty.

        arms: an ``ArmState`` (``wm.arms.new_state()``, one per rollout, frames in order) draws the robot arms from the
        kinematic robot model (one call of ``self.arms``): its pose is read from this latent and snapped onto the predicted grippers, the world
        model keeps only its gripper Gaussians and the rig's Gaussians are appended (cls 3, with a view-dependent ``sh1``)."""
        g, coarse = self.decode_raw(z)
        rig = boxes = None
        if arms is not None and g is not None:
            rig = self.arms(arms, coarse, g)
            boxes = [(arms.boxes[0][a], arms.boxes[1][a], arms.boxes[2][a]) for a in (0, 1)]
        if g is None:
            return None
        g.pop("rgb01_lock")
        x, cls = g["xyz"], g["cls"]
        in_boxes = None
        if boxes:
            from models.arm_rig import in_box

            in_boxes = in_box(x, *boxes[0]) | in_box(x, *boxes[1])
        keep = (g["scale"].max(1).values <= 0.02) & (g["opacity"] > 0.003)
        if boxes:
            keep &= (cls != 1) | in_boxes  # the rig draws the arm: robot gaussians survive only at the grippers
        if self.class_filters:  # table class floating above the table = mislabelled cables; small detached object pieces = floaters
            keep &= ~((cls == 2) & (x[:, 2] > TABLE_Z + 0.03)) & ~self._isolated(x, cls == 0)
        elif boxes and self.arms.cfg["table_float"]:  # the same cable filter where the class filters are off, except at the grippers
            keep &= ~((cls == 2) & (x[:, 2] > TABLE_Z + self.arms.cfg["table_float_z"]) & ~in_boxes)
        if self.table_feather > 0 or self.table_edge_faces:
            dface = torch.stack([x[:, 0] - self.bmin[0], self.bmax[0] - x[:, 0], x[:, 1] - self.bmin[1], self.bmax[1] - x[:, 1]], 1)
            if self.table_edge_faces:  # everything low in a band along a bbox face (a camera mount on the table edge), except at the grippers
                band = (dface < torch.tensor(self.table_edge_faces, device=x.device, dtype=x.dtype)).any(1) & (x[:, 2] < TABLE_Z + 0.08)
                keep &= ~(band if in_boxes is None else band & ~in_boxes)
            if self.table_feather > 0:  # the table layer fades out towards the bbox faces instead of ending on a line
                w = (dface.min(1).values / self.table_feather).clamp(0, 1)
                w = w * w * (3 - 2 * w)
                low = (cls == 2) & (x[:, 2] < TABLE_Z + 0.03)
                g["opacity"] = g["opacity"] * torch.where(low, w, torch.ones_like(w))
        keep = torch.nonzero(keep).squeeze(1)
        g = {k: v.index_select(0, keep) for k, v in g.items()}
        if rig is not None:
            n, m = g["xyz"].shape[0], rig["xyz"].shape[0]
            cls = torch.cat([g["cls"], torch.full((m,), 3, device=cls.device, dtype=cls.dtype)])
            g = {k: torch.cat([v, rig[k]]) for k, v in g.items() if k != "cls"}
            g["cls"] = cls
            g["sh1"] = torch.cat([torch.zeros(n, 3, 3, device=self.device), rig["sh1"]])
        return g

    @torch.no_grad()
    def tactile_particles(self, z):
        """Tactile particles of one latent, where the task has tactile pads: positions (N, 3) and contact signal (N,) in [0, 1].
        None for a model without the tactile particle head."""
        head = getattr(self.vae, "tactile_particle_head", None)
        if head is None:
            return None
        out = self.decode_cells(torch.as_tensor(z, device=self.device, dtype=torch.float32))
        occ, sig = out["tactile_occ_2"].float(), out["tactile_sig_2"].float()
        cells = torch.nonzero(occ[0, 0] > 0)
        if cells.shape[0] == 0:
            return torch.zeros(0, 3, device=self.device), torch.zeros(0, device=self.device)
        seeds = torch.cat([torch.zeros_like(cells[:, :1]), cells], 1)
        xyz, signal, keep = head(out["feat_2"].float(), torch.cat([occ, sig], 1), seeds, self.bmin.float(), self.vsize)
        keep = torch.sigmoid(keep.reshape(-1)) > 0.5
        return xyz.reshape(-1, 3)[keep].float(), torch.sigmoid(signal.reshape(-1))[keep].float()

    @torch.no_grad()
    def particles(self, z):
        """Occupied cells of the finest decoder level: centres (N, 3) and colours (N, 3) in [0, 1]."""
        z = torch.as_tensor(z, device=self.device, dtype=torch.float32)
        out = self.vae.decode(z[None], occ_mask_0=None, use_gt_mask=False)
        xyz, rgb = self.vae.decoder.extract_point_cloud(out["scene_occ_2"], out["rgb_2"], self.bmin, self.vae.spacing_fine)[0]
        return (xyz, (rgb.clamp(-1, 1) + 1) / 2)

    # -- cameras and rendering -----------------------------------------------------------------------------------------------
    def orbit_camera(self, az=None, el=None, radius=None, W=960, H=640, fov=50.0, pan=(0.0, 0.0, 0.0)):
        """A free camera on an orbit around the scene: viewmat (4, 4) world to camera and K (3, 3)."""
        v = self.cfg["view"]
        az, el, radius = (v["az"] if az is None else az, v["el"] if el is None else el, v["radius"] if radius is None else radius)
        target = ((self.bmin + self.bmax) / 2).cpu().numpy().astype(np.float64) + np.array([0.0, 0.0, -0.02]) + np.asarray(pan, np.float64)
        f = 0.5 * H / np.tan(np.deg2rad(fov) / 2)
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)
        return (torch.from_numpy(orbit_viewmat(target, az, el, radius)).to(self.device), torch.from_numpy(K).to(self.device), W, H)

    def episode_cameras(self, path=None, frame=0, cams=None, group="render_cams"):
        """The dataset cameras stored with a seed episode at ``frame``: list of (viewmat, K, W, H, camera id).
        group: ``render_cams`` (the training views) or ``latent_cams`` (the cameras recorded with the latents, where exported)."""
        path = path or self.dir / self.cfg["seed"]["file"]
        out = []
        with h5py.File(path, "r") as f:
            g = f[group]
            for c in cams or [int(x) for x in g.attrs["cam_ids"]]:
                gc = g[f"camera_{c}"]
                t = min(int(frame), gc["viewmat"].shape[0] - 1)
                vm = gc["viewmat"][t].astype(np.float32)
                if not np.isfinite(vm).all():
                    continue
                out.append((torch.from_numpy(vm).to(self.device), torch.from_numpy(gc["K"][t].astype(np.float32)).to(self.device), int(gc.attrs["W"]), int(gc.attrs["H"]), c))
        return out

    @torch.no_grad()
    def render(self, g, viewmat, K, W, H, *_, background=True):
        """Render gaussians over the scanned room (or over white): uint8 image (H, W, 3)."""
        if g is None or g["xyz"].shape[0] == 0:
            img = self.scan.render(viewmat, K, W, H)[0] if (background and self.scan is not None) else torch.ones(H, W, 3, device=self.device)
        elif background and self.scan is not None:
            fg, alpha, depth = render_gaussians(g["xyz"], g["scale"], g["opacity"], g["rgb01"], viewmat[None], K[None], W, H, quats=g["quat"], sh1=g.get("sh1"), with_depth=True)
            bg, bg_depth, bg_alpha = self.scan.render(viewmat, K, W, H)
            img = depth_composite(fg[0].clamp(0, 1), alpha[0], depth[0], bg, bg_depth, bg_alpha)
        else:
            fg, alpha = render_gaussians(g["xyz"], g["scale"], g["opacity"], g["rgb01"], viewmat[None], K[None], W, H, quats=g["quat"], sh1=g.get("sh1"))
            img = fg[0] + (1.0 - alpha[0])
        img = img.clamp(0, 1).mul(255).to(torch.uint8)
        if self.exposure != 1.0:  # display exposure in linear light, on the 8-bit frame as the public demo applies it
            img = (((img.float() / 255) ** 2.2 * self.exposure).clamp(0, 1) ** (1 / 2.2) * 255).round().to(torch.uint8)
        return img.cpu().numpy()

    @torch.no_grad()
    def render_points(self, xyz, rgb01, viewmat, K, W, H, *_, radius=0.0025):
        """Points drawn as opaque spheres over white: uint8 image (H, W, 3)."""
        n = xyz.shape[0]
        scale = torch.full((n, 3), radius, device=self.device)
        fg, alpha = render_gaussians(xyz, scale, torch.ones(n, device=self.device), rgb01, viewmat[None], K[None], W, H)
        return (fg[0] + (1.0 - alpha[0])).clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
