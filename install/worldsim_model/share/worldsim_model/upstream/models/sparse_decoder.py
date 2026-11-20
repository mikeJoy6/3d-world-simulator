"""The finest decoder level evaluated only near the surface, with the trained dense weights (inference speed-up).

The finest level (3x3 convolution on the parent grid, nearest x2 upsampling, two residual blocks, the z-mix convolution and
the occupancy head) is about 85 % of the decoding time, but only a few percent of its cells are occupied. Far from the scene
every activation equals one per-channel background vector, so each layer is carried as

    active cells (the children of occupied parent cells, dilated by a halo)  +  one background vector for all other cells.

Convolutions run on the active cells with spconv; the contribution of background neighbours and of the zero padding at the
grid border is added in closed form, and GroupNorm statistics are computed exactly over the full grid. The output matches
the dense decoder wherever the cells outside the active set really are background (render difference well under 1/255).
"""
import spconv.pytorch as spconv
import torch
import torch.nn.functional as F

K133 = [(0, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
K311 = [(dz, 0, 0) for dz in (-1, 0, 1)]
NEIGHBORS = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def taps(conv, offsets):
    """Conv3d weight (Cout, Cin, kz, ky, kx) as (K, Cin, Cout), one slice per kernel offset."""
    w = conv.weight
    kz, ky, kx = w.shape[2:]
    return torch.stack([w[:, :, dz + kz // 2, dy + ky // 2, dx + kx // 2].T for dz, dy, dx in offsets])


def norm_silu(x, bg, n_bg: int, weight, bias, groups: int, eps: float):
    """GroupNorm over the full grid (active rows x plus n_bg cells equal to bg), then SiLU.
    Returns the active rows and the new background."""
    n, c = x.shape
    cg = c // groups
    s1 = x.sum(0) + n_bg * bg
    s2 = (x * x).sum(0) + n_bg * bg * bg
    count = (n + n_bg) * cg
    mean = s1.view(groups, cg).sum(1) / count
    var = (s2.view(groups, cg).sum(1) / count - mean * mean).clamp(min=0)
    mean = mean.repeat_interleave(cg)
    scale = torch.rsqrt(var + eps).repeat_interleave(cg) * weight
    return F.silu((x - mean) * scale + bias), F.silu((bg - mean) * scale + bias)


def add_background(y, wb, outside, residual):
    """Sparse convolution rows plus the background taps (minus the taps that fall outside the grid) (plus a residual)."""
    bg = wb.sum(0)
    y = y.float() + bg - outside @ wb
    return (y if residual is None else y + residual), bg


class SparseFeatures:
    """The finest feature grid held sparsely: answers the Gaussian head's 3x3x3 neighbourhood gather without the dense grid."""

    def __init__(self, rows, bg, index, shape):
        self.rows, self.bg, self.index, self.shape = rows, bg, index, shape

    def float(self):
        return self

    def gather_nbrs(self, seeds, bf16=False):
        D, H, W = self.shape
        n = self.rows.shape[0]
        ext = torch.cat([self.rows, self.bg[None].to(self.rows.dtype), torch.zeros_like(self.bg)[None].to(self.rows.dtype)])
        o = torch.as_tensor(NEIGHBORS, device=seeds.device, dtype=seeds.dtype)
        zz, yy, xx = seeds[:, 1:2] + o[:, 0], seeds[:, 2:3] + o[:, 1], seeds[:, 3:4] + o[:, 2]
        inside = (zz >= 0) & (zz < D) & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        r = self.index[zz.clamp(0, D - 1), yy.clamp(0, H - 1), xx.clamp(0, W - 1)]
        r = torch.where(r < 0, torch.full_like(r, n), r)
        r = torch.where(inside, r, torch.full_like(r, n + 1))
        return ext[r] if bf16 else ext[r].float()


class SparseDecoder:
    """Drop-in for ``decoder(conv_decoder(z))`` at inference: same occupancy outputs, ``feat_2`` as SparseFeatures."""

    def __init__(self, decoder, halo=(1, 2, 2), compile=True):
        self.d, self.halo = decoder, halo
        f = decoder.finest
        up = decoder.up(f)
        assert len(up.upsample) == 2 and isinstance(up.upsample[1], torch.nn.Upsample)
        self.up_conv, self.blocks, self.zmix, occ = up.upsample[0], list(up.refine), decoder.final_z_mix, decoder.occ_head(f)
        self.W_up, self.b_up = taps(self.up_conv, K133).to(torch.bfloat16), self.up_conv.bias
        self.W_blk = [(taps(b.c1, K133).half(), taps(b.c2, K133).half()) for b in self.blocks]
        self.W_z, self.b_z = taps(self.zmix[0], K311).half(), self.zmix[0].bias
        self.W_occ, self.b_occ = occ.weight[:, :, 0, 0, 0].T.to(torch.bfloat16), occ.bias
        dev = self.W_up.device

        def sparse(conv, ks, key):
            m = spconv.SubMConv3d(conv.in_channels, conv.out_channels, ks, padding=tuple(k // 2 for k in ks), bias=False, indice_key=key).to(dev)
            with torch.no_grad():
                m.weight.copy_(conv.weight.permute(0, 2, 3, 4, 1))
            return m.half()

        self.sp_blk = [(sparse(b.c1, (1, 3, 3), "k133"), sparse(b.c2, (1, 3, 3), "k133")) for b in self.blocks]
        self.sp_z = sparse(self.zmix[0], (3, 1, 1), "k311")

        def parent(features):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f0 = decoder.level0_conv(features)
                o0 = decoder.occ_head(0)(f0).float()
                up0 = self.up_conv(f0).float()
            return o0, up0, f0[0, :, ::2, ::2, ::2].flatten(1).float().median(1).values

        self.parent = torch.compile(parent, dynamic=False) if compile else parent
        self.norm_silu = torch.compile(norm_silu, dynamic=True) if compile else norm_silu
        self.add_background = torch.compile(add_background, dynamic=True) if compile else add_background
        self.children = torch.tensor([(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)], device=dev)

    @torch.no_grad()
    def __call__(self, features):
        o0, up0, b0 = self.parent(features)
        _, _, Z0, Y0, X0 = o0.shape
        dev = o0.device
        hz, hy, hx = self.halo
        active = F.max_pool3d((o0[0, 0] > 0)[None, None].float(), (2 * hz + 1, 2 * hy + 1, 2 * hx + 1), 1, (hz, hy, hx))[0, 0] > 0
        P = torch.nonzero(active)
        xb = torch.einsum("c,kcd->d", b0.to(torch.bfloat16), self.W_up).float() + self.b_up
        x = up0[0][:, P[:, 0], P[:, 1], P[:, 2]].T.float().contiguous().repeat_interleave(8, 0)
        C = (P[:, None] * 2 + self.children[None]).reshape(-1, 3)
        n, shape = C.shape[0], (2 * Z0, 2 * Y0, 2 * X0)
        n_bg = shape[0] * shape[1] * shape[2] - n
        st = spconv.SparseConvTensor(
            torch.zeros(n, x.shape[1], dtype=torch.float16, device=dev), torch.cat([torch.zeros(n, 1, dtype=torch.int32, device=dev), C.int()], 1), list(shape), 1
        )

        def outside(offsets):
            nc = C[:, None] + torch.tensor(offsets, device=dev)[None]
            return ((nc[..., 0] < 0) | (nc[..., 0] >= shape[0]) | (nc[..., 1] < 0) | (nc[..., 1] >= shape[1]) | (nc[..., 2] < 0) | (nc[..., 2] >= shape[2])).float()

        out133, out311 = outside(K133), outside(K311)
        for blk, (W1, W2), (m1, m2) in zip(self.blocks, self.W_blk, self.sp_blk):
            h, hb = self.norm_silu(x, xb, n_bg, blk.n1.weight, blk.n1.bias, blk.n1.num_groups, blk.n1.eps)
            h, hb = self.add_background(m1(st.replace_feature((h - hb).half())).features, torch.einsum("c,kcd->kd", hb.half(), W1).float(), out133, None)
            h, hb = self.norm_silu(h, hb, n_bg, blk.n2.weight, blk.n2.bias, blk.n2.num_groups, blk.n2.eps)
            x, hb = self.add_background(m2(st.replace_feature((h - hb).half())).features, torch.einsum("c,kcd->kd", hb.half(), W2).float(), out133, x)
            xb = xb + hb
        y = self.sp_z(st.replace_feature((x - xb).half())).features
        y, yb = self.add_background(y, torch.einsum("c,kcd->kd", xb.half(), self.W_z).float(), out311, None)
        gn = self.zmix[1]
        y, yb = norm_silu(y + self.b_z, yb + self.b_z, n_bg, gn.weight, gn.bias, gn.num_groups, gn.eps)
        oa = (y.to(torch.bfloat16) @ self.W_occ).float() + self.b_occ
        ob = (yb.to(torch.bfloat16) @ self.W_occ).float() + self.b_occ
        occ = ob.view(1, -1, 1, 1, 1).expand(1, oa.shape[1], *shape).contiguous()
        occ[0, :, C[:, 0], C[:, 1], C[:, 2]] = oa.T
        index = torch.full(shape, -1, dtype=torch.long, device=dev)
        index[C[:, 0], C[:, 1], C[:, 2]] = torch.arange(n, device=dev)
        out = dict(feat_2=SparseFeatures(x.to(torch.bfloat16), xb, index, shape), zmix=(y, yb, C, shape))
        for i, o in ((0, o0), (2, occ)):
            out[f"scene_occ_{i}"], out[f"tac_occ_{i}"], out[f"signal_{i}"] = o[:, 0:1], o[:, 1:2], o[:, 2:3]
        if occ.shape[1] >= 5:
            out["tactile_occ_2"], out["tactile_sig_2"] = occ[:, 3:4], occ[:, 4:5]
        return out
