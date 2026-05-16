"""
Contrastive loss for EdgeMiner: symmetric InfoNCE (CLIP-style).

Given a batch of N camera-LiDAR pairs from the same N scenes, the loss treats
the diagonal of the similarity matrix as positive pairs and off-diagonal as
negatives. Symmetric means we compute both image-to-LiDAR and LiDAR-to-image
losses and average them.
"""

import torch
import torch.nn.functional as F


def info_nce_loss(
    image_emb: torch.Tensor,
    lidar_emb: torch.Tensor,
    logit_scale: torch.Tensor,
) -> dict:
    """
    Symmetric InfoNCE loss.

    Args:
        image_emb: (B, D) L2-normalized image embeddings
        lidar_emb: (B, D) L2-normalized LiDAR embeddings
        logit_scale: scalar temperature multiplier (exp of learnable parameter)

    Returns:
        dict with:
            loss: scalar tensor, mean of i2l and l2i cross-entropy
            acc_i2l: image-to-LiDAR top-1 accuracy (diagonal hit rate)
            acc_l2i: LiDAR-to-image top-1 accuracy
    """
    # Similarity matrix (B, B). Diagonal entries are positive pairs.
    logits = logit_scale * image_emb @ lidar_emb.t()

    # Targets: each row should match itself (same index in the batch)
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i2l = F.cross_entropy(logits, labels)
    loss_l2i = F.cross_entropy(logits.t(), labels)
    loss = (loss_i2l + loss_l2i) / 2

    # Top-1 accuracy for diagnostics
    with torch.no_grad():
        acc_i2l = (logits.argmax(dim=1) == labels).float().mean()
        acc_l2i = (logits.argmax(dim=0) == labels).float().mean()

    return {
        "loss": loss,
        "acc_i2l": acc_i2l.item(),
        "acc_l2i": acc_l2i.item(),
    }


def recall_at_k(
    image_emb: torch.Tensor,
    lidar_emb: torch.Tensor,
    k_values: tuple = (1, 5, 10),
) -> dict:
    """
    Compute Recall@K for image-to-LiDAR retrieval on a held-out set.

    For each image, rank all LiDAR embeddings by similarity. A retrieval counts
    as correct if the matching LiDAR (same index) is in the top-K.

    Args:
        image_emb: (N, D) embeddings for all val images
        lidar_emb: (N, D) embeddings for all val LiDAR scans
        k_values: which K values to report

    Returns:
        dict mapping "recall@K" -> float
    """
    sim = image_emb @ lidar_emb.t()  # (N, N)
    n = sim.size(0)
    labels = torch.arange(n, device=sim.device)

    # For each query (row), get the rank of the correct answer
    # argsort descending: higher similarity = lower rank index
    ranks = sim.argsort(dim=1, descending=True)

    results = {}
    for k in k_values:
        topk = ranks[:, :k]                                    # (N, k)
        hits = (topk == labels.unsqueeze(1)).any(dim=1).float()
        results[f"recall@{k}"] = hits.mean().item()
    return results
