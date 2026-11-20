"""The scanned room behind the world model.

The room was captured once as a Gaussian splat. It is rendered on its own for every camera and the world model's gaussians are
composited over it with a depth test, so a camera pole or a rail in the scan can stand in front of the predicted scene. The file
holds the splat already placed in the dataset's world frame: ``xyz`` (N, 3), ``scale`` (N, 3), ``opacity`` (N,), ``rgb01`` (N, 3) and
``quat`` (N, 4, wxyz).
"""
import torch
from gsplat import rasterization

LUM = (0.299, 0.587, 0.114)
TABLE_TOP = (-0.85, 0.75, -0.6, 0.55, 0.06)  # the scanned table top: x0, x1, y0, y1 (m) and half thickness about the table height
WALLS = ((1, 0.65), (0, 1.25))  # the scan's wall planes as (axis, position in m); gaussians within WALL_TOL of one are wall
WALL_TOL = 0.22


def outside_bbox(x, lo, hi, xy_margin=0.12, z_margin=0.05):
    """Scan gaussians that may occlude the model: further than ``xy_margin`` sideways from the model's box or ``z_margin`` above it.
    The scan's own table top never hides the model."""
    lo = torch.as_tensor(lo, device=x.device, dtype=x.dtype)
    hi = torch.as_tensor(hi, device=x.device, dtype=x.dtype)
    near = (x[:, 0] > lo[0] - xy_margin) & (x[:, 0] < hi[0] + xy_margin)
    near &= (x[:, 1] > lo[1] - xy_margin) & (x[:, 1] < hi[1] + xy_margin) & (x[:, 2] < hi[2] + z_margin)
    return ~near


class ScanBackground:

    def __init__(self, path, lo, hi, device, color_gain=None, color_bias=None, white_balance=None, room_gain=1.0, table_z=-0.076, wall_saturation=None, table_gain=None):
        """white_balance, wall_saturation and room_gain act on the room only (everything but the scanned table top);
        table_gain (r, g, b) brightens the scan's table top to the world model's own table layer, so the two meet without a step."""
        d = torch.load(path, map_location=device, weights_only=False)
        self.g = {k: d[k].to(device).float() for k in ("xyz", "scale", "opacity", "rgb01", "quat")}
        xyz = self.g["xyz"]
        if table_gain is not None:
            top = (xyz[:, 2] - table_z).abs() < 0.02
            self.g["rgb01"][top] = (self.g["rgb01"][top] * torch.as_tensor(table_gain, device=device, dtype=torch.float32)).clamp(0, 1)
        self.outside = outside_bbox(xyz, lo, hi).float()[:, None]
        self.table_z = table_z
        self.lo, self.hi, self._points = (torch.as_tensor(lo, device=device, dtype=torch.float32), torch.as_tensor(hi, device=device, dtype=torch.float32), None)
        x0, x1, y0, y1, dz = TABLE_TOP
        self.room = (~((xyz[:, 0] > x0) & (xyz[:, 0] < x1) & (xyz[:, 1] > y0) & (xyz[:, 1] < y1) & ((xyz[:, 2] - table_z).abs() < dz))).float()[:, None]
        wall = torch.zeros(xyz.shape[0], dtype=torch.bool, device=device)
        for axis, pos in WALLS:
            wall |= (xyz[:, axis] - pos).abs() < WALL_TOL
        self.wall = (wall & (xyz[:, 2] > table_z - 0.6)).float()[:, None]
        self.wall_saturation = None if wall_saturation is None else float(wall_saturation)
        self.gain = None if color_gain is None else torch.as_tensor(color_gain, device=device, dtype=torch.float32)
        self.bias = None if color_bias is None else torch.as_tensor(color_bias, device=device, dtype=torch.float32)
        self.wb = None if white_balance is None else torch.as_tensor(white_balance, device=device, dtype=torch.float32)
        self.room_gain = float(room_gain)
        self.lum = torch.tensor(LUM, device=device)
        self.cache = {}

    def _colors(self, rgb, room, wall):
        w = self.lum
        r = rgb
        if self.wb is not None:
            r = (rgb * (self.wb / (self.wb * w).sum())).clamp(0, 1)
        if self.wall_saturation is not None:  # the walls at their own saturation, at constant luminance
            l = (r * w).sum(-1, keepdim=True)
            r = r * (1 - wall) + (l + (r - l) * self.wall_saturation).clamp(0, 1) * wall
        rgb = rgb * (1 - room) + r * room
        if self.gain is not None:
            l0 = (rgb * w).sum(-1, keepdim=True)
            l1 = ((rgb * self.gain + self.bias).clamp(0, 1) * w).sum(-1, keepdim=True)
            rgb = (rgb * (l1 / l0.clamp_min(1e-3))).clamp(0, 1)
        if self.room_gain != 1.0:
            rgb = rgb * (1 - room) + ((rgb**2.2) * self.room_gain).clamp(0, 1) ** (1 / 2.2) * room
        return rgb

    @torch.no_grad()
    def points(self, grid=0.004, min_opacity=0.1, fill=12, gain=0.85, jitter=0.7):
        """The room as a point cloud of even density for the particle view: one point per ``grid`` metres, the table top densified
        with ``fill`` samples inside each Gaussian, colours as rendered and darkened by ``gain`` so a white table reads on white.
        Returns xyz (N, 3) and rgb (N, 3); computed once."""
        if self._points is None:
            g = self.g
            x, rgb, op = g["xyz"], self._colors(g["rgb01"], self.room, self.wall) * gain, g["opacity"].reshape(-1)
            if fill > 0:
                table = (x[:, 0] > self.lo[0] - 0.05) & (x[:, 0] < self.hi[0] + 0.05) & (x[:, 1] > self.lo[1] - 0.05) & (x[:, 1] < self.hi[1] + 0.05) & (x[:, 2] < self.table_z + 0.03)
                gen = torch.Generator(device=x.device).manual_seed(0)
                xt, sc = x[table], g["scale"][table].amax(1, keepdim=True).clamp(max=0.03)
                off = torch.randn(fill, xt.shape[0], 3, device=x.device, generator=gen) * sc[None]
                off[..., 2] *= 0.1  # spread along the surface, not off it
                x, rgb, op = torch.cat([x, (xt[None] + off).reshape(-1, 3)]), torch.cat([rgb, rgb[table].repeat(fill, 1)]), torch.cat([op, op[table].repeat(fill)])
            x, rgb = x[op >= min_opacity], rgb[op >= min_opacity]
            q = torch.floor(x / grid).long()
            q = q - q.amin(0, keepdim=True)
            key = (q[:, 0] * 2097152 + q[:, 1]) * 2097152 + q[:, 2]
            uq, inv = torch.unique(key, return_inverse=True)
            first = torch.full((len(uq),), x.shape[0], device=x.device, dtype=torch.long).scatter_reduce(0, inv, torch.arange(x.shape[0], device=x.device), "amin")
            u = torch.stack([((uq * m) % 1000003).float() / 1000003 for m in (2654435761, 40503, 97)], 1)  # hashed offset per cell: no lattice
            self._points = (x[first] + (u - 0.5) * (jitter * grid), rgb[first].clamp(0, 1))
        return self._points

    @torch.no_grad()
    def render(self, viewmat, K, W, H):
        """-> image (H, W, 3) in [0, 1], depth (H, W), and the weight (H, W) of scan content that may occlude the model."""
        key = (viewmat.cpu().numpy().tobytes(), K.cpu().numpy().tobytes(), W, H)
        if key not in self.cache:
            if len(self.cache) >= 8:
                self.cache.pop(next(iter(self.cache)))
            g = self.g
            colors = torch.cat([g["rgb01"], self.outside, self.room, self.wall], 1)
            r, alpha, _ = rasterization(
                means=g["xyz"], quats=g["quat"], scales=g["scale"], opacities=g["opacity"], colors=colors, viewmats=viewmat[None],
                Ks=K[None], width=W, height=H, sh_degree=None, near_plane=0.05, far_plane=10.0, rasterize_mode="antialiased",
                render_mode="RGB+ED",
            )
            a = alpha[0].clamp_min(0.02)
            im = (r[0, ..., :3] / a).clamp(0, 1)
            room, wall = (r[0, ..., 4:5] / a).clamp(0, 1), (r[0, ..., 5:6] / a).clamp(0, 1)
            self.cache[key] = (self._colors(im, room, wall), r[0, ..., 6], r[0, ..., 3].clamp(0, 1))
        return self.cache[key]
