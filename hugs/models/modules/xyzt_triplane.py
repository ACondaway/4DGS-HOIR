import torch
import torch.nn as nn
import torch.nn.functional as F

class XYZTTriPlane(nn.Module):
    def __init__(self, features=32, resX=256, resY=256, resZ=256, time_dim=10):
        super().__init__()
        self.plane_xy = nn.Parameter(torch.randn(1, features, resX, resY))
        self.plane_xz = nn.Parameter(torch.randn(1, features, resX, resZ))
        self.plane_yz = nn.Parameter(torch.randn(1, features, resY, resZ))
        self.plane_xt = nn.Parameter(torch.randn(1, features, resX, time_dim))
        self.plane_yt = nn.Parameter(torch.randn(1, features, resY, time_dim))
        self.plane_zt = nn.Parameter(torch.randn(1, features, resZ, time_dim))
        self.dim = features
        self.n_input_dims = 4  # Now we have an additional time dimension
        self.n_output_dims = 3 * features
        self.center = 0.0
        self.scale = 2.0

    def forward(self, x):
        x = (x - self.center) / self.scale + 0.5
        assert x.max() <= 1 + 1e-6 and x.min() >= -1e-6, f"x must be in [0, 1], got {x.min()} and {x.max()}"

        x = x * 2 - 1
        shape = x.shape
        coords = x.reshape(1, -1, 1, 3)
        time_coords = x.reshape(1, -1, 1, 1)  # Assuming time is the last dimension

        # Sample features from spatial planes
        feat_xy = F.grid_sample(self.plane_xy, coords[..., [0, 1]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_xz = F.grid_sample(self.plane_xz, coords[..., [0, 2]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_yz = F.grid_sample(self.plane_yz, coords[..., [1, 2]], align_corners=True)[0, :, :, 0].transpose(0, 1)

        # Sample features from temporal planes
        feat_xt = F.grid_sample(self.plane_xt, coords[..., [0]:], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_yt = F.grid_sample(self.plane_yt, coords[..., [1]:], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_zt = F.grid_sample(self.plane_zt, coords[..., [2]:], align_corners=True)[0, :, :, 0].transpose(0, 1)

        # Combine spatial and temporal features
        feat = torch.cat([feat_xy, feat_xz, feat_yz, feat_xt, feat_yt, feat_zt], dim=1)
        feat = feat.reshape(*shape[:-1], 6 * self.dim)  # 6 because we have features from 3 spatial and 3 temporal planes

        return feat
