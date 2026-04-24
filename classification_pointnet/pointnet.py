"""
pointnet.py — PointNet segmentation architecture.

PointNet processes raw 3D point clouds directly.
Each point is processed independently then global features
are aggregated and combined with local features for segmentation.

Reference: Qi et al., "PointNet: Deep Learning on Point Sets
for 3D Classification and Segmentation", CVPR 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """
    T-Net: learns a transformation matrix to align input points.
    Makes PointNet invariant to geometric transformations.
    """
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Initialize as identity matrix
        iden = torch.eye(self.k, requires_grad=True).repeat(
            batch_size, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()

        x = x.view(-1, self.k, self.k) + iden
        return x


class PointNetSeg(nn.Module):
    """
    PointNet for point cloud segmentation.

    WHY POINTNET?
    - Processes raw XYZ + features directly (no voxelization needed)
    - Permutation invariant (order of points doesn't matter)
    - Learns global + local features combined
    - Proven architecture for 3D point cloud classification

    Architecture:
        Input: (B, N, d_in) — batch of point clouds
        Local features per point → Global feature (max pool) →
        Combine local + global → Per-point classification
    """

    def __init__(self, d_in, num_classes=3):
        super(PointNetSeg, self).__init__()

        # Input transformation
        self.tnet1 = TNet(k=3)

        # Shared MLPs (implemented as 1D convolutions)
        self.conv1 = nn.Conv1d(d_in, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 512, 1)
        self.conv5 = nn.Conv1d(512, 2048, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)

        # Segmentation head — combines local + global features
        self.conv6  = nn.Conv1d(2048 + 128, 256, 1)
        self.conv7  = nn.Conv1d(256, 256, 1)
        self.conv8  = nn.Conv1d(256, 128, 1)
        self.conv9  = nn.Conv1d(128, num_classes, 1)

        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, xyz, features):
        """
        xyz      : (B, N, 3) — point coordinates
        features : (B, N, F) — per-point features

        Returns  : (B, N, num_classes) — per-point class scores
        """
        B, N, _ = xyz.shape

        # Concatenate xyz and features
        x = torch.cat([xyz, features], dim=-1)  # (B, N, 3+F)
        x = x.transpose(2, 1)                    # (B, 3+F, N)

        # Local feature extraction
        x  = F.relu(self.bn1(self.conv1(x)))     # (B, 64, N)
        x  = F.relu(self.bn2(self.conv2(x)))     # (B, 128, N)
        x  = F.relu(self.bn3(self.conv3(x)))     # (B, 128, N)
        local_feat = x                            # save for later

        x  = F.relu(self.bn4(self.conv4(x)))     # (B, 512, N)
        x  = F.relu(self.bn5(self.conv5(x)))     # (B, 2048, N)

        # Global feature — max pooling over all points
        global_feat = torch.max(x, 2, keepdim=True)[0]  # (B, 2048, 1)
        global_feat = global_feat.repeat(1, 1, N)        # (B, 2048, N)

        # Combine local + global
        x = torch.cat([local_feat, global_feat], dim=1)  # (B, 2048+128, N)

        # Segmentation head
        x = F.relu(self.bn6(self.conv6(x)))      # (B, 256, N)
        x = F.relu(self.bn7(self.conv7(x)))      # (B, 256, N)
        x = self.dropout(x)
        x = F.relu(self.bn8(self.conv8(x)))      # (B, 128, N)
        x = self.conv9(x)                         # (B, num_classes, N)

        x = x.transpose(2, 1)                    # (B, N, num_classes)
        return x
