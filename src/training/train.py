"""
Training loop for EdgeMiner dual encoder.

Designed to run end-to-end in Colab on a T4 GPU. Logs to TensorBoard and
saves the best checkpoint (by val Recall@5) to Google Drive.

Usage in Colab:
    from src.training.train import train_edgeminer
    train_edgeminer(
        dataroot='/content/data/nuscenes',
        checkpoint_dir='/content/drive/MyDrive/edgeminer/checkpoints',
        log_dir='/content/edgeminer_logs',
        num_epochs=20,
        batch_size=32,
        lr=3e-4,
    )

Or from CLI:
    python -m src.training.train --config configs/default.yaml
"""

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# These imports assume the file is run as part of the src package.
# In Colab where you paste the contents, the imports below will already be
# defined in the notebook namespace, so the try/except handles both cases.
try:
    from src.data.nuscenes_loader import NuScenesPairs
    from src.models.dual_encoder import DualEncoder
    from src.training.contrastive_loss import info_nce_loss, recall_at_k
except ImportError:
    pass  # Assume classes are already in the Colab namespace


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, val_ratio: float = 0.2, seed: int = 42):
    """80/20 train/val split by sample token (deterministic)."""
    rng = random.Random(seed)
    all_tokens = [s["token"] for s in dataset.nusc.sample]
    rng.shuffle(all_tokens)
    n_val = int(len(all_tokens) * val_ratio)
    val_tokens = set(all_tokens[:n_val])
    train_tokens = set(all_tokens[n_val:])

    train_ds = NuScenesPairs(
        dataroot=dataset.dataroot,
        version="v1.0-mini",
        split_tokens=list(train_tokens),
    )
    val_ds = NuScenesPairs(
        dataroot=dataset.dataroot,
        version="v1.0-mini",
        split_tokens=list(val_tokens),
    )
    return train_ds, val_ds


@torch.no_grad()
def evaluate(model, val_loader, device) -> dict:
    """Compute val loss and Recall@K over the full val set."""
    model.eval()
    all_image_emb, all_lidar_emb = [], []
    losses = []

    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        points = batch["lidar"].to(device, non_blocking=True)
        out = model(images, points)
        loss_dict = info_nce_loss(
            out["image_emb"], out["lidar_emb"], out["logit_scale"]
        )
        losses.append(loss_dict["loss"].item())
        all_image_emb.append(out["image_emb"])
        all_lidar_emb.append(out["lidar_emb"])

    image_emb = torch.cat(all_image_emb, dim=0)
    lidar_emb = torch.cat(all_lidar_emb, dim=0)
    recall = recall_at_k(image_emb, lidar_emb, k_values=(1, 5, 10))

    return {
        "val_loss": np.mean(losses),
        **recall,
        "num_val": image_emb.size(0),
    }


def train_edgeminer(
    dataroot: str,
    checkpoint_dir: str = "/content/drive/MyDrive/edgeminer/checkpoints",
    log_dir: str = "/content/edgeminer_logs",
    num_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 2,
    embed_dim: int = 256,
    freeze_image_backbone: bool = True,
    seed: int = 42,
    log_every: int = 5,
) -> dict:
    """
    Train the dual encoder end-to-end.

    Returns the final metrics dict from the best checkpoint.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ---- Data ----
    print("Loading nuScenes mini...")
    full_ds = NuScenesPairs(dataroot=dataroot, version="v1.0-mini")
    train_ds, val_ds = split_dataset(full_ds, val_ratio=0.2, seed=seed)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ---- Model ----
    model = DualEncoder(
        embed_dim=embed_dim,
        freeze_image_backbone=freeze_image_backbone,
    ).to(device)

    # Only optimize parameters that require gradients (skip frozen DINOv2)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ---- Logging & checkpointing setup ----
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")
    print(f"Checkpoints will save to: {checkpoint_dir}")

    # ---- Baseline (random init) eval ----
    baseline = evaluate(model, val_loader, device)
    print(f"\nBaseline (random projection): {baseline}")
    writer.add_scalar("val/recall@5", baseline["recall@5"], 0)
    writer.add_scalar("val/recall@1", baseline["recall@1"], 0)

    # ---- Training loop ----
    best_recall5 = baseline["recall@5"]
    best_metrics = baseline
    step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses, epoch_accs = [], []

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            points = batch["lidar"].to(device, non_blocking=True)

            optimizer.zero_grad()
            out = model(images, points)
            loss_dict = info_nce_loss(
                out["image_emb"], out["lidar_emb"], out["logit_scale"]
            )
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_accs.append((loss_dict["acc_i2l"] + loss_dict["acc_l2i"]) / 2)

            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/logit_scale", out["logit_scale"].item(), step)
            step += 1

        scheduler.step()
        train_loss = np.mean(epoch_losses)
        train_acc = np.mean(epoch_accs)

        # ---- Validation ----
        val_metrics = evaluate(model, val_loader, device)
        writer.add_scalar("val/loss", val_metrics["val_loss"], epoch)
        writer.add_scalar("val/recall@1", val_metrics["recall@1"], epoch)
        writer.add_scalar("val/recall@5", val_metrics["recall@5"], epoch)
        writer.add_scalar("val/recall@10", val_metrics["recall@10"], epoch)
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("train/acc_epoch", train_acc, epoch)

        print(
            f"Epoch {epoch:2d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"R@1={val_metrics['recall@1']:.3f} "
            f"R@5={val_metrics['recall@5']:.3f} "
            f"R@10={val_metrics['recall@10']:.3f}"
        )

        # ---- Checkpoint best model by val Recall@5 ----
        if val_metrics["recall@5"] > best_recall5:
            best_recall5 = val_metrics["recall@5"]
            best_metrics = {**val_metrics, "epoch": epoch}
            ckpt_path = Path(checkpoint_dir) / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": val_metrics,
                },
                ckpt_path,
            )
            print(f"  ✓ New best Recall@5={best_recall5:.3f} — saved to {ckpt_path}")

    writer.close()
    print(f"\nTraining complete. Best metrics: {best_metrics}")
    return best_metrics


if __name__ == "__main__":
    # Default Colab paths
    train_edgeminer(
        dataroot="/content/data/nuscenes",
        checkpoint_dir="/content/drive/MyDrive/edgeminer/checkpoints",
        log_dir="/content/edgeminer_logs",
        num_epochs=20,
        batch_size=32,
        lr=3e-4,
    )
