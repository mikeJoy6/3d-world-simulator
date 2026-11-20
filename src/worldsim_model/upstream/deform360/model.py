"""Deform360 dynamics: a coarse 3D U-Net whose action is drawn into the voxel grid.

Each gripper is a Gaussian blob at its contact point in the 32^3 latent grid, plus the same blob scaled by the measured
opening: four action channels per frame (two per arm). They are concatenated to the latent channels of every observed frame
and given on their own for the frame to predict. The full pose vector is also embedded at the coarse 8^3 level, where causal
attention runs over time.

Observed frames are one reference frame (any earlier frame of the same segment) followed by five history frames.

    action (23,) = per arm [contact point xyz (m, grid frame), first two rows of the rotation (6), opening (m)], two arm-present
                   flags, frame stride
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.dynamics_coarse3d import Block, TimeBlock

N_HISTORY, ACTION_DIM, GRID = 5, 23, 32
BOUNDS = ((-0.025, -0.415, -0.1), (0.49, 0.11, 0.28))  # the latent grid in the packed z-up frame, metres
SIGMA = 1.5  # width of a gripper blob, voxels


class SpatialCoarse(nn.Module):

    def __init__(self, width=176):
        super().__init__()
        w, c = width, 4 * width
        self.enc0 = Block(6 * 13 + 4, w)  # 6 observed frames of 8 latent + 4 action + 1 noise channels, and the target's 4 action channels
        self.enc1, self.mid = Block(w, 2 * w), Block(2 * w, c)
        self.pose = nn.Sequential(nn.Linear(ACTION_DIM + 1, 128), nn.SiLU(), nn.Linear(128, c))
        self.noise = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, c))
        self.lat = nn.Conv3d(13, c, 1)
        self.temporal = nn.Sequential(TimeBlock(c), TimeBlock(c))
        self.ref_embed = nn.Parameter(torch.zeros(c))
        self.dec1, self.dec0, self.out = Block(6 * w, 2 * w), Block(3 * w, w), nn.Conv3d(w, 8, 1)
        self.register_buffer("action_mean", torch.zeros(ACTION_DIM))
        self.register_buffer("action_std", torch.ones(ACTION_DIM))
        lo, hi = torch.tensor(BOUNDS[0]), torch.tensor(BOUNDS[1])
        zz, yy, xx = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), torch.arange(GRID), indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy, zz], -1).float())
        self.register_buffer("lo", lo)
        self.register_buffer("vs", (hi - lo) / GRID)
        self.register_buffer("rot", torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]))

    def set_action_norm(self, mean, std):
        self.action_mean.copy_(torch.as_tensor(mean, device=self.action_mean.device))
        self.action_std.copy_(torch.as_tensor(std, device=self.action_std.device))

    def spatial_action(self, normalized):
        """(N, 23) normalised actions -> (N, 4, 32, 32, 32): per arm the contact-point blob and the blob times the opening."""
        a = normalized.float() * self.action_std + self.action_mean
        maps = []
        for arm in range(2):
            coord = (a[:, arm * 10 : arm * 10 + 3] - self.lo) / self.vs - 0.5
            dist = ((self.grid[None] - coord[:, None, None, None]) ** 2).sum(-1)
            heat = torch.exp(-dist / (2 * SIGMA**2)) * a[:, 20 + arm, None, None, None]
            maps += [heat, heat * a[:, arm * 10 + 9, None, None, None] / 0.1]
        return torch.stack(maps, 1)

    @torch.no_grad()
    def gripper_region(self, normalized, radius=4):
        """(N, 23) -> (N, 32, 32, 32) bool: a (2 radius + 1)^3 box of cells around each present gripper, for the loss."""
        a = normalized.float() * self.action_std + self.action_mean
        region = torch.zeros((len(a), GRID, GRID, GRID), device=a.device, dtype=torch.bool)
        for arm in range(2):
            center = torch.floor((a[:, arm * 10 : arm * 10 + 3] - self.lo) / self.vs)
            region |= ((self.grid[None] - center[:, None, None, None]).abs() <= radius).all(-1) & (a[:, 20 + arm, None, None, None] > 0.5)
        return region

    def forward(self, z_seq, actions, noise_std, anchor, anchor_action):
        """z_seq (B, 6, 8, 32, 32, 32): five history latents and the target slot (its content is ignored);
        actions (B, 6, 23) and noise_std (B, 6) for those slots; anchor (B, 1, 8, 32, 32, 32) with anchor_action (B, 1, 23).
        Returns the predicted latent (B, 8, 32, 32, 32)."""
        b = len(z_seq)
        observed = torch.cat([anchor, z_seq[:, :5]], 1)
        poses = torch.cat([anchor_action, actions], 1)
        noises = torch.cat([noise_std.new_zeros(b, 1), noise_std], 1)
        maps = self.spatial_action(poses.reshape(-1, ACTION_DIM)).reshape(b, 7, 4, GRID, GRID, GRID)
        level = noises[:, :6, None, None, None, None].expand(-1, -1, 1, GRID, GRID, GRID)
        observed = torch.cat([observed, maps[:, :6].to(observed.dtype), level.to(observed.dtype)], 2)
        x0 = self.enc0(torch.cat([observed.flatten(1, 2), maps[:, -1].to(observed.dtype)], 1))
        x1 = self.enc1(F.avg_pool3d(x0, 2))
        x2 = self.mid(F.avg_pool3d(x1, 2))
        c = x2.shape[1]
        stride = poses[:, -1, -1] * self.action_std[-1] + self.action_mean[-1]
        times = z_seq.new_tensor([0, -4, -3, -2, -1, 0, 1])[None] * stride[:, None] / 30
        emb = self.pose(torch.cat([poses, times[..., None]], -1)) + self.noise(noises[..., None])
        emb = emb + torch.cat([self.ref_embed[None, None].expand(b, 1, -1), emb.new_zeros(b, 6, c)], 1)
        low = self.lat(F.adaptive_avg_pool3d(observed.flatten(0, 1), (8, 8, 8))).reshape(b, 6, c, 8, 8, 8)
        tokens = torch.cat([low, low.new_zeros(b, 1, c, 8, 8, 8)], 1) + x2[:, None] + emb[..., None, None, None]
        tokens = self.temporal(tokens.permute(0, 3, 4, 5, 1, 2).reshape(-1, 7, c))
        future = tokens.reshape(b, 8, 8, 8, 7, c)[:, :, :, :, -1].permute(0, 4, 1, 2, 3)

        def decode(f, s1, s0):
            y = self.dec1(torch.cat([F.interpolate(f, size=(16, 16, 16), mode="trilinear", align_corners=False), s1], 1))
            return self.out(self.dec0(torch.cat([F.interpolate(y, size=(32, 32, 32), mode="trilinear", align_corners=False), s0], 1)))

        return checkpoint(decode, future, x1, x0, use_reentrant=False) if self.training else decode(future, x1, x0)
