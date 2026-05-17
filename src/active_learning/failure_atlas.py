"""
Failure Atlas for EdgeMiner.

Builds the headline portfolio artifact: a 2-D UMAP projection of all scene
embeddings, colored by detector uncertainty. The "failure halo" — high-uncertainty
scenes — clusters visibly in embedding space, revealing the systematic failure
modes of the perception model.

Combined with FAISS retrieval from Phase 2, this enables the full active learning
workflow: pick a failure scene, retrieve similar failures via FAISS, treat the
cluster as a labeling target.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import umap
except ImportError:
    raise ImportError("UMAP not installed. In Colab: !pip install umap-learn -q")


def build_failure_atlas(
    embeddings_path: str,
    uncertainty_scores: dict,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> dict:
    """
    Project embeddings to 2-D via UMAP and pair with uncertainty scores.

    Args:
        embeddings_path: path to embeddings .pt file from Phase 2
        uncertainty_scores: dict of {sample_token: uncertainty_dict}
        n_neighbors, min_dist: UMAP hyperparameters
        random_state: for reproducibility

    Returns:
        dict with keys:
            coords_2d:  (N, 2) array of UMAP positions
            tokens:     list of N sample tokens (same order as coords_2d)
            composite:  (N,) array of composite uncertainty scores
            num_detections: (N,) array of detection counts
    """
    data = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    image_emb = data["image_embeddings"].numpy()
    tokens = data["sample_tokens"]

    print(f"Fitting UMAP on {image_emb.shape[0]} embeddings...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric="cosine",  # right choice for L2-normalized embeddings
    )
    coords_2d = reducer.fit_transform(image_emb)
    print(f"UMAP done. Shape: {coords_2d.shape}")

    composite = np.array([uncertainty_scores[t]["composite_score"] for t in tokens])
    num_dets = np.array([uncertainty_scores[t]["num_detections"] for t in tokens])

    return {
        "coords_2d": coords_2d,
        "tokens": tokens,
        "composite": composite,
        "num_detections": num_dets,
    }


def plot_failure_atlas(
    atlas: dict,
    save_path: Optional[str] = None,
    title: str = "EdgeMiner Failure Atlas",
    figsize: tuple = (12, 9),
) -> plt.Figure:
    """
    Render the headline scatter plot: UMAP coords colored by uncertainty.

    Highlights the top-10% most uncertain scenes (the failure halo).
    """
    coords = atlas["coords_2d"]
    composite = atlas["composite"]

    # Top 10% threshold for "failure" scenes
    threshold = np.percentile(composite, 90)
    is_failure = composite >= threshold

    fig, ax = plt.subplots(figsize=figsize)

    # Plot easy scenes first (background), then failures on top
    sc = ax.scatter(
        coords[~is_failure, 0],
        coords[~is_failure, 1],
        c=composite[~is_failure],
        cmap="viridis",
        s=40,
        alpha=0.6,
        edgecolors="none",
        label=f"Confident scenes ({(~is_failure).sum()})",
    )
    ax.scatter(
        coords[is_failure, 0],
        coords[is_failure, 1],
        c="red",
        s=80,
        alpha=0.9,
        edgecolors="black",
        linewidths=1,
        label=f"Top-10% uncertain ({is_failure.sum()})",
    )

    plt.colorbar(sc, ax=ax, label="Uncertainty score (composite)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved atlas to: {save_path}")

    return fig


def find_failure_clusters(
    atlas: dict,
    index,                            # EdgeMinerIndex from Phase 2
    uncertainty_scores: dict,
    top_n_failures: int = 5,
    cluster_size: int = 5,
) -> list:
    """
    For each of the top-N most uncertain scenes, retrieve K similar scenes
    via FAISS. Result is a list of "failure clusters" suitable for batch
    labeling — the active learning output.

    Returns:
        list of dicts, one per failure cluster:
            {
                "anchor_token": str,
                "anchor_score": float,
                "anchor_idx": int,
                "cluster_tokens": list of K tokens (anchor + K-1 retrieved),
                "cluster_indices": list of K dataset indices,
                "cluster_similarities": list of K similarity scores,
            }
    """
    # Rank scenes by uncertainty (descending)
    ranked = sorted(
        uncertainty_scores.items(),
        key=lambda x: x[1]["composite_score"],
        reverse=True,
    )[:top_n_failures]

    clusters = []
    for token, scores in ranked:
        anchor_idx = atlas["tokens"].index(token)
        # Retrieve cluster_size similar scenes (including anchor itself)
        results = index.query_image_to_image(anchor_idx, k=cluster_size)

        clusters.append({
            "anchor_token": token,
            "anchor_score": scores["composite_score"],
            "anchor_idx": anchor_idx,
            "anchor_num_detections": scores["num_detections"],
            "cluster_tokens": [r["sample_token"] for r in results],
            "cluster_indices": [r["index"] for r in results],
            "cluster_similarities": [r["similarity"] for r in results],
        })

    return clusters


def visualize_failure_cluster(
    cluster: dict,
    nusc,
    dataroot: str,
    save_path: Optional[str] = None,
    camera_channel: str = "CAM_FRONT",
) -> plt.Figure:
    """
    Render one failure cluster as a row of camera images.

    Anchor on the left, retrieved similar failures to the right.
    This is the per-cluster screenshot for the README.
    """
    from PIL import Image

    k = len(cluster["cluster_tokens"])
    fig, axes = plt.subplots(1, k, figsize=(3 * k, 3))

    for i, (token, sim) in enumerate(zip(cluster["cluster_tokens"], cluster["cluster_similarities"])):
        sample = nusc.get("sample", token)
        cam_data = nusc.get("sample_data", sample["data"][camera_channel])
        img = Image.open(f"{dataroot}/{cam_data['filename']}")
        axes[i].imshow(img)
        if i == 0:
            axes[i].set_title(
                f"FAILURE ANCHOR\nuncertainty={cluster['anchor_score']:.2f}\n"
                f"detections={cluster['anchor_num_detections']}",
                fontsize=9, color="red", fontweight="bold",
            )
        else:
            axes[i].set_title(f"#{i}  sim={sim:.3f}", fontsize=9)
        axes[i].axis("off")

    plt.suptitle("Failure Cluster: anchor + 4 similar scenes (active learning batch)",
                 fontsize=11)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved cluster viz to: {save_path}")

    return fig
