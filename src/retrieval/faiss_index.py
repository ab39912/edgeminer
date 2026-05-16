"""
FAISS index wrapper for EdgeMiner retrieval.

Supports:
    - Building image and LiDAR indices from saved embeddings
    - Cross-modal queries (image -> LiDAR, LiDAR -> image)
    - Within-modal queries (image -> image, "find scenes that look like this")
    - Persistence to disk

For 404 samples we use a simple exact index (IndexFlatIP for cosine similarity
on L2-normalized vectors). For larger datasets you'd swap to IndexIVFPQ or HNSW.

Usage:
    from src.retrieval.faiss_index import EdgeMinerIndex

    index = EdgeMinerIndex.from_embeddings_file(
        '/content/drive/MyDrive/edgeminer/embeddings/all.pt'
    )
    # Query: given image #5, find the 5 most similar LiDAR scenes
    results = index.query_image_to_lidar(query_idx=5, k=5)
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS not installed. In Colab: !pip install faiss-cpu -q"
    )


class EdgeMinerIndex:
    """
    Holds FAISS indices for image and LiDAR embeddings, supports cross-modal search.

    Because embeddings are L2-normalized, inner product equals cosine similarity.
    We use IndexFlatIP for exact search (fine for thousands of samples).
    """

    def __init__(
        self,
        image_embeddings: np.ndarray,
        lidar_embeddings: np.ndarray,
        sample_tokens: list,
        metadata: Optional[dict] = None,
    ):
        assert image_embeddings.shape == lidar_embeddings.shape, \
            f"Shape mismatch: {image_embeddings.shape} vs {lidar_embeddings.shape}"
        assert len(sample_tokens) == image_embeddings.shape[0], \
            f"Token count {len(sample_tokens)} != embedding count {image_embeddings.shape[0]}"

        self.image_embeddings = image_embeddings.astype(np.float32)
        self.lidar_embeddings = lidar_embeddings.astype(np.float32)
        self.sample_tokens = sample_tokens
        self.metadata = metadata or {}

        d = image_embeddings.shape[1]
        # IndexFlatIP = inner product. L2-normalized -> cosine similarity.
        self.image_index = faiss.IndexFlatIP(d)
        self.lidar_index = faiss.IndexFlatIP(d)
        self.image_index.add(self.image_embeddings)
        self.lidar_index.add(self.lidar_embeddings)

        print(f"EdgeMinerIndex built: {len(sample_tokens)} samples, dim={d}")

    @classmethod
    def from_embeddings_file(cls, path: str) -> "EdgeMinerIndex":
        """Load from the .pt file produced by extract_embeddings()."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            image_embeddings=data["image_embeddings"].numpy(),
            lidar_embeddings=data["lidar_embeddings"].numpy(),
            sample_tokens=data["sample_tokens"],
            metadata=data.get("metadata"),
        )

    # -------- Querying --------

    def _query(
        self,
        query_vec: np.ndarray,
        target_index: "faiss.Index",
        k: int,
    ) -> list:
        """Run a k-NN search and return list of dicts with token + similarity score."""
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        query_vec = query_vec.astype(np.float32)

        similarities, indices = target_index.search(query_vec, k)
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            results.append({
                "sample_token": self.sample_tokens[idx],
                "similarity": float(sim),
                "index": int(idx),
            })
        return results

    def query_image_to_lidar(self, query_idx: int, k: int = 5) -> list:
        """Given an image (by index), retrieve top-K matching LiDAR scans."""
        return self._query(self.image_embeddings[query_idx], self.lidar_index, k)

    def query_lidar_to_image(self, query_idx: int, k: int = 5) -> list:
        """Given a LiDAR scan (by index), retrieve top-K matching images."""
        return self._query(self.lidar_embeddings[query_idx], self.image_index, k)

    def query_image_to_image(self, query_idx: int, k: int = 5) -> list:
        """Find visually similar scenes in the image modality."""
        return self._query(self.image_embeddings[query_idx], self.image_index, k)

    def query_lidar_to_lidar(self, query_idx: int, k: int = 5) -> list:
        """Find scenes with similar 3D structure."""
        return self._query(self.lidar_embeddings[query_idx], self.lidar_index, k)

    def query_with_vector(
        self,
        query_vec: Union[np.ndarray, torch.Tensor],
        target_modality: str = "image",
        k: int = 5,
    ) -> list:
        """
        Query with an arbitrary embedding vector (e.g., from a new image).

        Args:
            query_vec: (D,) or (1, D) embedding
            target_modality: "image" or "lidar" — which index to search
            k: number of results
        """
        if isinstance(query_vec, torch.Tensor):
            query_vec = query_vec.detach().cpu().numpy()
        target = self.image_index if target_modality == "image" else self.lidar_index
        return self._query(query_vec, target, k)

    # -------- Persistence --------

    def save(self, dir_path: str):
        """Save FAISS indices + embeddings to disk."""
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.image_index, str(d / "image.faiss"))
        faiss.write_index(self.lidar_index, str(d / "lidar.faiss"))
        torch.save({
            "image_embeddings": torch.from_numpy(self.image_embeddings),
            "lidar_embeddings": torch.from_numpy(self.lidar_embeddings),
            "sample_tokens": self.sample_tokens,
            "metadata": self.metadata,
        }, d / "embeddings.pt")
        print(f"Saved index to: {dir_path}")

    @classmethod
    def load(cls, dir_path: str) -> "EdgeMinerIndex":
        d = Path(dir_path)
        return cls.from_embeddings_file(str(d / "embeddings.pt"))


if __name__ == "__main__":
    # Build the index from an existing embeddings file
    EMBED_PATH = "/content/drive/MyDrive/edgeminer/embeddings/all.pt"
    INDEX_DIR = "/content/drive/MyDrive/edgeminer/index"

    index = EdgeMinerIndex.from_embeddings_file(EMBED_PATH)

    # Quick sanity check: query with sample 0, see if it retrieves itself
    results = index.query_image_to_lidar(query_idx=0, k=3)
    print("\nQuery: image #0 -> top-3 LiDAR matches")
    for r in results:
        print(f"  idx={r['index']:3d}  sim={r['similarity']:.3f}  token={r['sample_token'][:8]}...")

    index.save(INDEX_DIR)
