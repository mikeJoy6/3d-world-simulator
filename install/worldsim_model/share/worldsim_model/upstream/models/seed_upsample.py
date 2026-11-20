import torch

NEIGHBOR_OFFSETS = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def gather_neighborhood_features(feat_grid, seed_indices):
    B, C, D, H, W = feat_grid.shape
    S = seed_indices.shape[0]
    if B == 1:
        z = seed_indices[:, 1]
        y = seed_indices[:, 2]
        x = seed_indices[:, 3]
        offs = torch.as_tensor(NEIGHBOR_OFFSETS, device=feat_grid.device, dtype=seed_indices.dtype)
        zz = z[:, None] + offs[:, 0]
        yy = y[:, None] + offs[:, 1]
        xx = x[:, None] + offs[:, 2]
        inb = (zz >= 0) & (zz < D) & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        lin = ((zz.clamp(0, D - 1).long() * H + yy.clamp(0, H - 1).long()) * W + xx.clamp(0, W - 1).long()).reshape(-1)
        flat = feat_grid.reshape(1, C, D * H * W)
        out = flat.gather(2, lin.view(1, 1, -1).expand(1, C, -1))
        nbr_feats = out.view(C, S, 27).permute(1, 2, 0)
        return nbr_feats * inb.unsqueeze(-1).to(nbr_feats.dtype)
    raise NotImplementedError("gather_neighborhood_features: batch size must be 1")


def local_chamfer_loss(
    seed_indices,
    part_world_xyz,
    part_rgb,
    gt_xyz_world,
    gt_rgb,
    bounds_min,
    voxel_size,
    fine_shape,
    rgb_weight=0.1,
    kernel=3,
    seed_weights=None,
    bwd_weight=1.0,
    return_components=False,
):
    device = part_world_xyz.device
    S, P, _ = part_world_xyz.shape
    if S == 0:
        z = torch.zeros((), device=device)
        return (z, z, z, z) if return_components else z
    B, N, _ = gt_xyz_world.shape
    D, H, W = fine_shape
    n_vox = D * H * W
    dhw = H * W
    normed = (gt_xyz_world - bounds_min) / voxel_size
    ix = normed[..., 0].long().clamp(0, W - 1)
    iy = normed[..., 1].long().clamp(0, H - 1)
    iz = normed[..., 2].long().clamp(0, D - 1)
    gt_flat_local = iz * dhw + iy * W + ix
    batch_idx_bn = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)
    gt_flat_global = (batch_idx_bn * n_vox + gt_flat_local).reshape(-1)
    all_gt_xyz = gt_xyz_world.reshape(-1, 3)
    all_gt_rgb = gt_rgb.reshape(-1, 3)
    sort_perm = gt_flat_global.argsort()
    gt_flat_sorted = gt_flat_global[sort_perm]
    gt_xyz_sorted = all_gt_xyz[sort_perm]
    gt_rgb_sorted = all_gt_rgb[sort_perm]
    voxel_counts = torch.bincount(gt_flat_sorted, minlength=B * n_vox)
    voxel_ends = voxel_counts.cumsum(0)
    voxel_starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), voxel_ends[:-1]])
    seed_b = seed_indices[:, 0]
    seed_z = seed_indices[:, 1]
    seed_y = seed_indices[:, 2]
    seed_x = seed_indices[:, 3]
    assert kernel % 2 == 1 and kernel >= 1, "kernel must be odd"
    r = kernel // 2
    ncount = kernel**3
    kr = range(-r, r + 1)
    offsets = torch.tensor([(dz, dy, dx) for dz in kr for dy in kr for dx in kr], device=device, dtype=torch.long)
    nz = (seed_z.unsqueeze(1) + offsets[:, 0]).clamp(0, D - 1)
    ny = (seed_y.unsqueeze(1) + offsets[:, 1]).clamp(0, H - 1)
    nx = (seed_x.unsqueeze(1) + offsets[:, 2]).clamp(0, W - 1)
    nbr_local = nz * dhw + ny * W + nx
    nbr_global = seed_b.unsqueeze(1) * n_vox + nbr_local
    nbr_global_flat = nbr_global.reshape(-1)
    starts_flat = voxel_starts[nbr_global_flat]
    ends_flat = voxel_ends[nbr_global_flat]
    lengths_flat = ends_flat - starts_flat
    total_T = int(lengths_flat.sum().item())
    if total_T == 0:
        z = torch.zeros((), device=device)
        return (z, z, z, z) if return_components else z
    segment_ids = torch.repeat_interleave(torch.arange(S * ncount, device=device), lengths_flat)
    cum_starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), lengths_flat.cumsum(0)[:-1]])
    within = torch.arange(total_T, device=device) - cum_starts[segment_ids]
    gt_sorted_idx = starts_flat[segment_ids] + within
    gt_xyz_per_entry = gt_xyz_sorted[gt_sorted_idx]
    gt_rgb_per_entry = gt_rgb_sorted[gt_sorted_idx]
    seed_id_per_entry = segment_ids // ncount
    pred_xyz_per_entry = part_world_xyz[seed_id_per_entry]
    part_rgb[seed_id_per_entry]
    dists = (gt_xyz_per_entry.unsqueeze(1) - pred_xyz_per_entry).norm(dim=-1)
    flat_key = seed_id_per_entry.unsqueeze(1) * P + torch.arange(P, device=device).unsqueeze(0)
    fwd_min = torch.full((S * P,), float("inf"), device=device)
    fwd_min.scatter_reduce_(0, flat_key.reshape(-1), dists.reshape(-1), reduce="amin", include_self=True)
    fwd_min = fwd_min.view(S, P)
    bwd_min_per_entry, bwd_min_idx = dists.min(dim=1)
    bwd_sum = torch.zeros(S, device=device).scatter_add(0, seed_id_per_entry, bwd_min_per_entry)
    bwd_count = torch.zeros(S, device=device).scatter_add(0, seed_id_per_entry, torch.ones_like(bwd_min_per_entry))
    valid = bwd_count > 0
    fwd_per_seed = fwd_min.mean(dim=1)
    bwd_per_seed = bwd_sum / bwd_count.clamp(min=1)
    cd_per_seed = fwd_per_seed + bwd_weight * bwd_per_seed
    if valid.sum() == 0:
        z = torch.zeros((), device=device)
        return (z, z, z, z) if return_components else z
    if seed_weights is None:
        cd_loss = cd_per_seed[valid].mean()
    else:
        w = seed_weights.to(cd_per_seed.dtype)[valid]
        cd_loss = (cd_per_seed[valid] * w).sum() / w.sum().clamp(min=1.0)
    glob_pid = seed_id_per_entry * P + bwd_min_idx
    tgt_sum = torch.zeros(S * P, 3, device=device).scatter_add(0, glob_pid.unsqueeze(1).expand(-1, 3), gt_rgb_per_entry)
    tgt_cnt = torch.zeros(S * P, device=device).scatter_add(0, glob_pid, torch.ones_like(bwd_min_per_entry))
    has = tgt_cnt > 0
    if has.any():
        tgt_rgb = tgt_sum[has] / tgt_cnt[has].unsqueeze(1)
        pred_flat = part_rgb.reshape(S * P, 3)[has]
        rgb_l1_pp = (pred_flat - tgt_rgb).abs().mean(dim=-1)
        if seed_weights is None:
            rgb_loss = rgb_l1_pp.mean()
        else:
            seed_of_p = (torch.arange(S * P, device=device) // P)[has]
            w = seed_weights.to(rgb_l1_pp.dtype)[seed_of_p]
            rgb_loss = (rgb_l1_pp * w).sum() / w.sum().clamp(min=1.0)
    else:
        rgb_loss = torch.zeros((), device=device)
    if not return_components:
        return cd_loss + rgb_weight * rgb_loss
    fwd_diag = fwd_per_seed[valid].mean().detach()
    bwd_diag = bwd_per_seed[valid].mean().detach()
    return (cd_loss + rgb_weight * rgb_loss, fwd_diag, bwd_diag, rgb_loss.detach())


def two_round_chamfer_loss(
    seed_indices,
    head_out,
    gt_xyz,
    gt_rgb,
    bounds_min,
    voxel_size,
    fine_shape,
    rgb_weight=0.1,
    kernel=3,
    seed_weights=None,
    round2_weight=1.0,
    bwd_weight=1.0,
):
    r1_xyz, r1_rgb, r2_xyz, r2_rgb = head_out
    if r1_xyz.shape[0] == 0:
        z = torch.zeros((), device=r1_xyz.device)
        return (z, z, z, z, z, z)
    cd1, f1, b1, g1 = local_chamfer_loss(
        seed_indices,
        r1_xyz,
        r1_rgb,
        gt_xyz,
        gt_rgb,
        bounds_min,
        voxel_size,
        fine_shape=fine_shape,
        rgb_weight=rgb_weight,
        kernel=kernel,
        seed_weights=seed_weights,
        bwd_weight=bwd_weight,
        return_components=True,
    )
    cd2, f2, b2, g2 = local_chamfer_loss(
        seed_indices,
        r2_xyz,
        r2_rgb,
        gt_xyz,
        gt_rgb,
        bounds_min,
        voxel_size,
        fine_shape=fine_shape,
        rgb_weight=rgb_weight,
        kernel=kernel,
        seed_weights=seed_weights,
        bwd_weight=bwd_weight,
        return_components=True,
    )
    return (cd1 + round2_weight * cd2, cd1, cd2, f1 + f2, b1 + b2, g1 + g2)

