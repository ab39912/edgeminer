"""
Extract embeddings from a trained EdgeMiner checkpoint over the full dataset.

Output: a single .pt file containing:
    - image_embeddings: (N, D) tensor
    - lidar_embeddings: (N, D) tensor
    - sample_tokens: list of N strings
    - metadata: dict (checkpoint path, model config, dataset version)

Usage in Colab (with NuScenesPairs and DualEncoder already in namespace):
    from src.retrieval.embedding_extractor import extract_embeddings
    extract_embeddings(
        checkpoint_path='/content/drive/MyDrive/edgeminer/checkpoints/best.pt',
        dataroot='/content/data/nuscenes',
        output_path='/content/drive/MyDrive/edgeminer/embeddings/all.pt',
    )
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from src.data.nuscenes_loader import NuScenesPairs
    from src.models.dual_encoder import DualEncoder
except ImportError:
    pass  # Assume classes are in Colab namespace


@torch.no_grad()
def extract_embeddings(
    checkpoint_path: str,
    dataroot: str,
    output_path: str,
    version: str = "v1.0-mini",
    batch_size: int = 32,
    num_workers: int = 2,
    embed_dim: int = 256,
    device: Optional[str] = None,
) -> dict:
    """
    Run the trained dual encoder over every sample and save embeddings.

    Returns the same dict that gets saved to disk.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Extracting embeddings on: {device}")

    # ---- Load checkpoint ----
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = DualEncoder(embed_dim=embed_dim, freeze_image_backbone=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  Checkpoint trained for {ckpt.get('epoch', '?')} epochs")
    if "metrics" in ckpt:
        m = ckpt["metrics"]
        print(f"  Val metrics at this checkpoint: "
              f"R@1={m.get('recall@1', 0):.3f}, "
              f"R@5={m.get('recall@5', 0):.3f}, "
              f"R@10={m.get('recall@10', 0):.3f}")

    # ---- Load full dataset (no train/val split — we embed everything) ----
    dataset = NuScenesPairs(dataroot=dataroot, version=version)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # CRITICAL: preserve order so tokens line up with embeddings
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"Extracting embeddings for {len(dataset)} samples...")

    # ---- Forward pass over everything ----
    all_image_emb, all_lidar_emb, all_tokens = [], [], []

    for batch in tqdm(loader, desc="Encoding"):
        images = batch["image"].to(device, non_blocking=True)
        points = batch["lidar"].to(device, non_blocking=True)
        out = model(images, points)
        all_image_emb.append(out["image_emb"].cpu())
        all_lidar_emb.append(out["lidar_emb"].cpu())
        all_tokens.extend(batch["token"])

    image_emb = torch.cat(all_image_emb, dim=0)
    lidar_emb = torch.cat(all_lidar_emb, dim=0)

    print(f"\nDone. Image embeddings: {image_emb.shape}, LiDAR: {lidar_emb.shape}")
    print(f"All embeddings L2-normalized: "
          f"img_norm={image_emb.norm(dim=-1).mean().item():.4f}, "
          f"lidar_norm={lidar_emb.norm(dim=-1).mean().item():.4f}")

    # ---- Save ----
    output = {
        "image_embeddings": image_emb,
        "lidar_embeddings": lidar_emb,
        "sample_tokens": all_tokens,
        "metadata": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_metrics": ckpt.get("metrics"),
            "dataset_version": version,
            "embed_dim": embed_dim,
            "num_samples": len(all_tokens),
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"Saved to: {output_path}")

    return output


if __name__ == "__main__":
    extract_embeddings(
        checkpoint_path="/content/drive/MyDrive/edgeminer/checkpoints/best.pt",
        dataroot="/content/data/nuscenes",
        output_path="/content/drive/MyDrive/edgeminer/embeddings/all.pt",
    )
