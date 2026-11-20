"""The latent and voxel views of the demo: occupied grid cells drawn as shaded cubes by a GPU voxel ray-caster (one Triton kernel).

  latents: the coarse occupancy (the latent grid itself), each cell coloured by the first principal component of its latent
           vector, binned into four blues. The component is a fixed linear map fitted once per task (``latent_pca`` in task.json).
  voxels:  the finest occupancy in its decoded colour: the cells the Gaussians are generated from.
"""
import numpy as np
import torch
import triton
import triton.language as tl

PALETTE = torch.tensor([[8, 81, 156], [49, 130, 189], [107, 174, 214], [198, 219, 239]], dtype=torch.float32) / 255.0
K133 = torch.tensor([(0, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])


@triton.jit
def _cast(occ_ptr, rgb_ptr, out_ptr, Rm_ptr, prm_ptr, W, H, NX, NY, NZ, MAXS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0); offs = pid * BLOCK + tl.arange(0, BLOCK); valid = offs < W * H
    u = (offs % W).to(tl.float32) + 0.5; v = (offs // W).to(tl.float32) + 0.5
    fx = tl.load(prm_ptr + 0); fy = tl.load(prm_ptr + 1); cx = tl.load(prm_ptr + 2); cy = tl.load(prm_ptr + 3)
    ox = tl.load(prm_ptr + 4); oy = tl.load(prm_ptr + 5); oz = tl.load(prm_ptr + 6)              # camera centre, grid units
    gap = tl.load(prm_ptr + 7)                                                                # box inset per side (cell units)
    f0 = tl.abs(tl.load(prm_ptr + 8)); f1 = tl.abs(tl.load(prm_ptr + 9)); f2 = tl.abs(tl.load(prm_ptr + 10))   # camera forward (world): the light
    a = (u - cx) / fx; b = (v - cy) / fy
    # Rm = diag(1/spacing) @ R_c2w: camera ray -> grid-unit direction
    dx = tl.load(Rm_ptr + 0) * a + tl.load(Rm_ptr + 1) * b + tl.load(Rm_ptr + 2)
    dy = tl.load(Rm_ptr + 3) * a + tl.load(Rm_ptr + 4) * b + tl.load(Rm_ptr + 5)
    dz = tl.load(Rm_ptr + 6) * a + tl.load(Rm_ptr + 7) * b + tl.load(Rm_ptr + 8)
    nxf = NX.to(tl.float32); nyf = NY.to(tl.float32); nzf = NZ.to(tl.float32)
    BIG = 1e30
    ix_ = tl.where(dx != 0, 1.0 / dx, BIG); iy_ = tl.where(dy != 0, 1.0 / dy, BIG); iz_ = tl.where(dz != 0, 1.0 / dz, BIG)
    t1 = (0.0 - ox) * ix_; t2 = (nxf - ox) * ix_; t3 = (0.0 - oy) * iy_; t4 = (nyf - oy) * iy_; t5 = (0.0 - oz) * iz_; t6 = (nzf - oz) * iz_
    tmn = tl.maximum(tl.maximum(tl.minimum(t1, t2), tl.minimum(t3, t4)), tl.minimum(t5, t6)); tmn = tl.maximum(tmn, 0.0)
    tmx = tl.minimum(tl.minimum(tl.maximum(t1, t2), tl.maximum(t3, t4)), tl.maximum(t5, t6))
    alive = valid & (tmx > tmn)
    # entry axis = the slab whose near plane is tmn (shading of a hit in the first cell)
    ax = tl.where(tl.minimum(t1, t2) >= tmn - 1e-6, 0, tl.where(tl.minimum(t3, t4) >= tmn - 1e-6, 1, 2))
    px = ox + dx * (tmn + 1e-4); py = oy + dy * (tmn + 1e-4); pz = oz + dz * (tmn + 1e-4)
    cxi = tl.minimum(tl.maximum(tl.floor(px), 0.0), nxf - 1).to(tl.int32)
    cyi = tl.minimum(tl.maximum(tl.floor(py), 0.0), nyf - 1).to(tl.int32)
    czi = tl.minimum(tl.maximum(tl.floor(pz), 0.0), nzf - 1).to(tl.int32)
    sx = tl.where(dx > 0, 1, -1); sy = tl.where(dy > 0, 1, -1); sz = tl.where(dz > 0, 1, -1)
    tdx = tl.abs(ix_); tdy = tl.abs(iy_); tdz = tl.abs(iz_)
    tmxx = tl.where(dx > 0, (cxi.to(tl.float32) + 1.0 - ox) * ix_, tl.where(dx < 0, (cxi.to(tl.float32) - ox) * ix_, BIG))
    tmxy = tl.where(dy > 0, (cyi.to(tl.float32) + 1.0 - oy) * iy_, tl.where(dy < 0, (cyi.to(tl.float32) - oy) * iy_, BIG))
    tmxz = tl.where(dz > 0, (czi.to(tl.float32) + 1.0 - oz) * iz_, tl.where(dz < 0, (czi.to(tl.float32) - oz) * iz_, BIG))
    tcur = tmn; hit = tl.zeros([BLOCK], dtype=tl.int1); hidx = tl.zeros([BLOCK], dtype=tl.int32); hax = ax; ht = tmn
    for _ in range(MAXS):
        inside = alive & (cxi >= 0) & (cxi < NX) & (cyi >= 0) & (cyi < NY) & (czi >= 0) & (czi < NZ)
        idx = (czi * NY + cyi) * NX + cxi
        o = tl.load(occ_ptr + idx, mask=inside, other=0)
        # occupied cell: intersect the inset box (extent 1 - 2*gap, as the robo02 0.94 boxes); a ray through the gap marches on
        lx = cxi.to(tl.float32) + gap; ly = cyi.to(tl.float32) + gap; lz = czi.to(tl.float32) + gap; w = 1.0 - 2.0 * gap
        ax0 = (lx - ox) * ix_; ax1 = (lx + w - ox) * ix_; ay0 = (ly - oy) * iy_; ay1 = (ly + w - oy) * iy_; az0 = (lz - oz) * iz_; az1 = (lz + w - oz) * iz_
        tnx = tl.minimum(ax0, ax1); tny = tl.minimum(ay0, ay1); tnz = tl.minimum(az0, az1)
        tn = tl.maximum(tl.maximum(tnx, tny), tnz); tf = tl.minimum(tl.minimum(tl.maximum(ax0, ax1), tl.maximum(ay0, ay1)), tl.maximum(az0, az1))
        bax = tl.where((tnx >= tny) & (tnx >= tnz), 0, tl.where(tny >= tnz, 1, 2))
        nh = inside & (o != 0) & (tn <= tf) & (tf > 0)
        hit = hit | nh; hidx = tl.where(nh, idx, hidx); hax = tl.where(nh, bax, hax); ht = tl.where(nh, tn, ht)
        alive = inside & ~nh
        # step to the next cell along the axis with the nearest boundary
        mx = (tmxx <= tmxy) & (tmxx <= tmxz); my = (~mx) & (tmxy <= tmxz); mz = (~mx) & (~my)
        tcur = tl.where(mx, tmxx, tl.where(my, tmxy, tmxz)); ax = tl.where(mx, 0, tl.where(my, 1, 2))
        cxi = tl.where(mx, cxi + sx, cxi); cyi = tl.where(my, cyi + sy, cyi); czi = tl.where(mz, czi + sz, czi)
        tmxx = tl.where(mx, tmxx + tdx, tmxx); tmxy = tl.where(my, tmxy + tdy, tmxy); tmxz = tl.where(mz, tmxz + tdz, tmxz)
        alive = alive & (tcur < tmx)
    r = tl.load(rgb_ptr + hidx * 3 + 0, mask=hit, other=1.0); g = tl.load(rgb_ptr + hidx * 3 + 1, mask=hit, other=1.0); bb = tl.load(rgb_ptr + hidx * 3 + 2, mask=hit, other=1.0)
    # pyrender's look (scene_for in the robo02 script): ambient 0.65 + a directional light along the camera axis (intensity 1.5 x
    # Lambert of the default 0.2-metal material = 0.367), linear vertex colours, then gamma 1/2.2. Matches the preview's table / front faces.
    ndl = tl.where(hax == 0, f0, tl.where(hax == 1, f1, f2))
    k = 0.65 + 0.367 * ndl
    r = tl.where(hit, tl.exp(tl.log(tl.maximum(r * k, 1e-6)) / 2.2), 1.0); g = tl.where(hit, tl.exp(tl.log(tl.maximum(g * k, 1e-6)) / 2.2), 1.0); bb = tl.where(hit, tl.exp(tl.log(tl.maximum(bb * k, 1e-6)) / 2.2), 1.0)
    r = tl.minimum(r, 1.0); g = tl.minimum(g, 1.0); bb = tl.minimum(bb, 1.0)
    tl.store(out_ptr + offs * 3 + 0, r, mask=valid); tl.store(out_ptr + offs * 3 + 1, g, mask=valid); tl.store(out_ptr + offs * 3 + 2, bb, mask=valid)


def render_cells(occ, rgb, bmin, bmax, viewmat, K, W, H, gap=0.03, srgb=False, ss=2):
    """occ (nz,ny,nx) bool, rgb (nz,ny,nx,3) float 0..1 (linear like pyrender's vertex colours; srgb=True linearises first, for real
    colours), world bounds of the grid -> (H,W,3) uint8 image (white background). Cells drawn as boxes of extent 1 - 2*gap;
    ss x ss supersampling keeps the (sub-pixel) gaps as smooth thin lines instead of aliased dashes.
    viewmat world->camera 4x4 (OpenCV: +z forward), K 3x3 intrinsics."""
    dev = occ.device; nz, ny, nx = occ.shape
    bmin = torch.as_tensor(bmin, dtype=torch.float64).cpu().numpy(); bmax = torch.as_tensor(bmax, dtype=torch.float64).cpu().numpy()
    sp = (bmax - bmin) / np.array([nx, ny, nz], np.float64)
    V = np.asarray(viewmat, np.float64); Rc2w = V[:3, :3].T; c = -Rc2w @ V[:3, 3]
    Rm = (Rc2w / sp[:, None]).astype(np.float32); og = (c - bmin) / sp
    K = np.asarray(K, np.float64).copy(); K[:2] *= ss; W, H = W * ss, H * ss
    fwd = Rc2w[:, 2]
    prm = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], og[0], og[1], og[2], gap, fwd[0], fwd[1], fwd[2]], np.float32)
    if srgb: rgb = rgb.float().clamp(0, 1) ** 2.2
    out = torch.empty(H * W * 3, device=dev, dtype=torch.float32)
    BLOCK = 256
    _cast[(triton.cdiv(W * H, BLOCK),)](occ.contiguous().to(torch.uint8), rgb.reshape(-1, 3).float().contiguous(), out,
                                        torch.from_numpy(Rm.reshape(-1)).to(dev), torch.from_numpy(prm).to(dev), W, H, nx, ny, nz,
                                        MAXS=nx + ny + nz + 3, BLOCK=BLOCK)
    im = out.view(H, W, 3)
    if ss > 1: im = torch.nn.functional.avg_pool2d(im.permute(2, 0, 1)[None], ss)[0].permute(1, 2, 0)
    return (im * 255).round().clamp(0, 255).to(torch.uint8)


@torch.no_grad()
def sparse_colors(head, y, yb, C, shape, occ):
    """The decoder's colour head (3x3 convolution, SiLU, 1x1 convolution, tanh) at the occupied cells only, from the sparse
    decoder's rows y (N, C) at cells C (N, 3) and their background row yb -> colours (nz, ny, nx, 3) in [0, 1]."""
    dev = y.device
    c1, c2 = head[0], head[2]
    oc = occ.nonzero()
    n, S = y.shape[0], torch.tensor(shape, device=dev)
    index = torch.full(tuple(shape), -1, dtype=torch.long, device=dev)
    index[C[:, 0], C[:, 1], C[:, 2]] = torch.arange(n, device=dev)
    nc = oc[:, None] + K133.to(dev)[None]
    outside = ((nc < 0) | (nc >= S)).any(-1)
    nc = torch.minimum(nc.clamp(min=0), S - 1)
    r = index[nc[..., 0], nc[..., 1], nc[..., 2]]
    r = torch.where(outside, n + 1, torch.where(r < 0, n, r))  # inactive cell: background row; outside the grid: zero row
    ext = torch.cat([y.float(), yb.float()[None], torch.zeros_like(yb, dtype=torch.float32)[None]])
    W1 = torch.stack([c1.weight[:, :, 0, dy + 1, dx + 1].T for _, dy, dx in K133.tolist()]).float()
    h = torch.nn.functional.silu(torch.einsum("mkc,kcd->md", ext[r], W1) + c1.bias.float())
    rgb = torch.tanh(h @ c2.weight[:, :, 0, 0, 0].T.float() + c2.bias.float())
    out = torch.ones(*occ.shape, 3, device=dev)
    out[oc[:, 0], oc[:, 1], oc[:, 2]] = ((rgb + 1) / 2).clamp(0, 1)
    return out


@torch.no_grad()
def cells(wm, z, kind):
    """Occupied cells and their colours for one latent: occ (nz, ny, nx) bool and rgb (nz, ny, nx, 3)."""
    z = torch.as_tensor(z, device=wm.device, dtype=torch.float32)
    out = wm.decode_cells(z)
    if kind == "latents":
        P = wm.cfg["latent_pca"]
        c0, c1 = P["ch"]
        t = lambda k: torch.tensor(P[k], dtype=torch.float32, device=wm.device)
        s = torch.einsum("czyx,c->zyx", z[c0:c1] - t("mean")[:, None, None, None], t("pc"))
        return out["scene_occ_0"][0, 0] > 0, PALETTE.to(wm.device).flip(0)[torch.bucketize(s, t("q")[1:-1])]
    occ = out["scene_occ_2"][0, 0] > 0
    if "zmix" in out:
        return occ, sparse_colors(wm.vae.decoder.rgb_head, *out["zmix"], occ)
    return occ, ((out["rgb_2"][0].float().permute(1, 2, 3, 0) + 1) / 2).clamp(0, 1)
