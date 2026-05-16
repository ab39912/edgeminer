"""
Dual encoder for EdgeMiner: camera tower + LiDAR tower with shared embedding space.

Camera tower: DINOv2 (frozen backbone, trainable projection head)
LiDAR tower: lightweight PointNet (trainable end-to-end)

Both project to a shared L2-normalized embedding space for contrastive training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """DINOv2 ViT-S/14 backbone with a small projection head."""

    def __init__(self, embed_dim: int = 256, freeze_backbone: bool = True):
        super().__init__()
        # DINOv2 small, pretrained, self-supervised — great fit for the JD narrative
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        backbone_dim = 384  # ViT-S output dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # DINOv2 expects 14-pixel multiples — resize input from 224 to 224 (16*14) is fine
        features = self.backbone(images)  # (B, 384)
        emb = self.projection(features)   # (B, embed_dim)
        return F.normalize(emb, dim=-1)


class PointNetEncoder(nn.Module):
    """
    Minimal PointNet for LiDAR. Permutation-invariant via max pooling.

    Input: (B, N, 4) — N points with (x, y, z, intensity)
    Output: (B, embed_dim), L2 normalized
    """

    def __init__(self, embed_dim: int = 256, input_channels: int = 4):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: (B, N, 4) -> (B, 4, N) for Conv1d
        x = points.transpose(1, 2)
        x = self.point_mlp(x)            # (B, 512, N)
        x = torch.max(x, dim=2).values   # (B, 512), permutation-invariant
        emb = self.projection(x)         # (B, embed_dim)
        return F.normalize(emb, dim=-1)


class DualEncoder(nn.Module):
    """
    Wraps the camera and LiDAR towers into a single module.

    forward() returns a dict with both embeddings for use in InfoNCE loss.
    """

    def __init__(self, embed_dim: int = 256, freeze_image_backbone: bool = True):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, freeze_backbone=freeze_image_backbone)
        self.lidar_encoder = PointNetEncoder(embed_dim=embed_dim)
        # Learnable temperature, CLIP-style. Starts at 0.07 (in log space: ln(1/0.07) ≈ 2.66).
        self.logit_scale = nn.Parameter(torch.tensor(2.66))

    def forward(self, images: torch.Tensor, points: torch.Tensor) -> dict:
        return {
            "image_emb": self.image_encoder(images),
            "lidar_emb": self.lidar_encoder(points),
            "logit_scale": self.logit_scale.exp().clamp(max=100.0),
        }


def info_nce_loss(image_emb: torch.Tensor, lidar_emb: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """
    Symmetric InfoNCE loss (CLIP-style).

    Diagonal of the similarity matrix is positive pairs (same sample),
    off-diagonal is negatives (different samples in the batch).
    """
    logits = logit_scale * image_emb @ lidar_emb.t()        # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2l = F.cross_entropy(logits, labels)
    loss_l2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2l + loss_l2i) / 2


if __name__ == "__main__":
    # Sanity check with random inputs — run before training to catch shape bugs
    model = DualEncoder(embed_dim=256, freeze_image_backbone=True)
    B, N = 4, 4096
    fake_images = torch.randn(B, 3, 224, 224)
    fake_points = torch.randn(B, N, 4)

    out = model(fake_images, fake_points)
    print(f"Image embedding shape: {out['image_emb'].shape}")  # (4, 256)
    print(f"LiDAR embedding shape: {out['lidar_emb'].shape}")  # (4, 256)
    print(f"Logit scale: {out['logit_scale'].item():.3f}")

    loss = info_nce_loss(out["image_emb"], out["lidar_emb"], out["logit_scale"])
    print(f"InfoNCE loss (random init): {loss.item():.4f}")    # expect ~ln(B) = 1.39
