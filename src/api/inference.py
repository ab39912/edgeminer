"""
Inference pipeline for EdgeMiner.

Loads a trained dual encoder checkpoint and provides:
    - Image preprocessing (resize, normalize, batch)
    - Single-image and batch embedding extraction
    - Index loading and retrieval

Designed to be importable from FastAPI, Streamlit, scripts, and tests alike.
"""

import io
import time
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    from src.models.dual_encoder import DualEncoder
    from src.retrieval.faiss_index import EdgeMinerIndex
except ImportError:
    pass  # Resolved at runtime when modules are co-located


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class EdgeMinerInference:
    """
    Production-ready inference wrapper around the trained dual encoder.

    Typical usage:
        engine = EdgeMinerInference(
            checkpoint_path='best.pt',
            index_dir='index/',
            device='cpu',  # or 'cuda'
        )
        # Single query
        emb = engine.encode_image(pil_image)
        results = engine.search(emb, k=5)

        # Batch query
        embs = engine.encode_images_batch([img1, img2, img3])
    """

    def __init__(
        self,
        checkpoint_path: str,
        index_dir: Optional[str] = None,
        device: str = "cpu",
        image_size: int = 224,
        embed_dim: int = 256,
        quantized: bool = False,
    ):
        self.device = torch.device(device)
        self.embed_dim = embed_dim
        self.quantized = quantized

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # Optional: load FAISS index for retrieval endpoints
        self.index = None
        if index_dir is not None and Path(index_dir).exists():
            self.index = EdgeMinerIndex.load(index_dir)
            print(f"FAISS index loaded: {len(self.index.sample_tokens)} scenes")

    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model = DualEncoder(embed_dim=self.embed_dim, freeze_image_backbone=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(self.device)
        print(f"Model loaded from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
        return model

    # ---------- Preprocessing ----------

    def preprocess_image(self, image: Union[Image.Image, bytes, str]) -> torch.Tensor:
        """
        Accept PIL, raw bytes, or a file path. Return (3, H, W) tensor.

        This is the preprocessing layer the JD calls "scalable data preprocessing."
        """
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        return self.transform(image)

    # ---------- Encoding ----------

    @torch.no_grad()
    def encode_image(self, image: Union[Image.Image, bytes, str]) -> np.ndarray:
        """Encode a single image to a 256-d L2-normalized embedding."""
        tensor = self.preprocess_image(image).unsqueeze(0).to(self.device)
        emb = self.model.image_encoder(tensor)
        return emb.cpu().numpy().squeeze(0)

    @torch.no_grad()
    def encode_images_batch(
        self,
        images: List[Union[Image.Image, bytes, str]],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Batch inference for efficiency.

        Returns (N, 256) array. Always processes in chunks of `batch_size` so
        memory stays bounded for large inputs.
        """
        all_embs = []
        for i in range(0, len(images), batch_size):
            chunk = images[i:i + batch_size]
            tensors = torch.stack([self.preprocess_image(img) for img in chunk]).to(self.device)
            embs = self.model.image_encoder(tensors)
            all_embs.append(embs.cpu().numpy())
        return np.concatenate(all_embs, axis=0)

    # ---------- Retrieval ----------

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        target_modality: str = "image",
    ) -> List[dict]:
        """Retrieve top-K similar scenes from the FAISS index."""
        if self.index is None:
            raise RuntimeError("No FAISS index loaded. Pass index_dir at init.")
        return self.index.query_with_vector(
            query_embedding, target_modality=target_modality, k=k
        )

    # ---------- Diagnostics ----------

    def benchmark(self, num_images: int = 100, batch_size: int = 16) -> dict:
        """Measure throughput: images per second under realistic load."""
        # Create dummy data
        dummy_image = Image.new("RGB", (1600, 900), color=(128, 128, 128))
        images = [dummy_image] * num_images

        # Warmup (compile JIT paths, allocate memory)
        _ = self.encode_images_batch(images[:batch_size], batch_size=batch_size)

        # Timed run
        start = time.perf_counter()
        _ = self.encode_images_batch(images, batch_size=batch_size)
        elapsed = time.perf_counter() - start

        return {
            "num_images": num_images,
            "batch_size": batch_size,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_images_per_sec": round(num_images / elapsed, 2),
            "ms_per_image": round(elapsed / num_images * 1000, 3),
            "device": str(self.device),
            "quantized": self.quantized,
        }
