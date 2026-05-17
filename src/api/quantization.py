"""
Quantization for EdgeMiner.

Implements two quantization strategies and compares them:

    1. Dynamic quantization: weights quantized at load time, activations
       quantized at runtime. Simple, no calibration needed. Best for models
       dominated by Linear layers (which our projection heads are).

    2. Static quantization: weights AND activations quantized ahead of time
       using calibration data. Requires more setup but produces faster models.

Both are CPU-only in PyTorch's standard quantization toolkit. GPU quantization
exists (e.g. TensorRT, bitsandbytes) but is out of scope here.

The benchmark function compares all three (FP32, dynamic INT8, static INT8) on
the same workload and reports throughput + accuracy.
"""

import copy
import time
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    from src.api.inference import EdgeMinerInference
except ImportError:
    pass


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """
    Dynamic INT8 quantization of Linear layers.

    The projection heads in our DualEncoder are pure Linear stacks, so dynamic
    quantization captures most of the speedup available.
    """
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},   # quantize only Linear modules
        dtype=torch.qint8,
    )
    return quantized


def apply_static_quantization(
    model: nn.Module,
    calibration_images: List,
    inference_engine,
    backend: str = "fbgemm",
) -> nn.Module:
    """
    Static INT8 quantization with calibration.

    Args:
        model: a DualEncoder (we'll quantize only the image tower for the API)
        calibration_images: ~32-128 representative images to set activation ranges
        inference_engine: an EdgeMinerInference instance for preprocessing
        backend: 'fbgemm' for x86, 'qnnpack' for ARM

    Note: PyTorch's static quantization needs the model to declare quant/dequant
    stubs explicitly. For the DINOv2 ViT backbone this is non-trivial and we
    keep the backbone in FP32 for static. Only the projection head is quantized.
    """
    torch.backends.quantized.engine = backend
    model = copy.deepcopy(model).eval()

    # We wrap the projection head only — backbone stays FP32
    projection = model.image_encoder.projection

    # Insert quantization stubs
    class QuantizedProjection(nn.Module):
        def __init__(self, projection):
            super().__init__()
            self.quant = torch.quantization.QuantStub()
            self.projection = projection
            self.dequant = torch.quantization.DeQuantStub()

        def forward(self, x):
            x = self.quant(x)
            x = self.projection(x)
            x = self.dequant(x)
            return x

    wrapped = QuantizedProjection(projection).eval()
    wrapped.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(wrapped, inplace=True)

    # Calibrate by running representative data through the wrapped module
    print(f"Calibrating static quantization on {len(calibration_images)} images...")
    with torch.no_grad():
        for img in calibration_images:
            tensor = inference_engine.preprocess_image(img).unsqueeze(0)
            features = model.image_encoder.backbone(tensor)
            _ = wrapped(features)

    torch.quantization.convert(wrapped, inplace=True)
    model.image_encoder.projection = wrapped
    return model


def benchmark_all_modes(
    checkpoint_path: str,
    index_dir: Optional[str] = None,
    num_warmup: int = 5,
    num_timed: int = 100,
    batch_size: int = 16,
) -> dict:
    """
    Run all three modes on identical workload and compare.

    Returns a dict with throughput, latency, and embedding-similarity to FP32
    (the accuracy proxy).
    """
    results = {}

    # ---- FP32 baseline ----
    engine_fp32 = EdgeMinerInference(
        checkpoint_path=checkpoint_path, index_dir=None, device="cpu"
    )
    results["fp32"] = engine_fp32.benchmark(
        num_images=num_timed, batch_size=batch_size
    )

    # Reference embedding on a fixed input — we'll compare quantized embeddings to this
    ref_image = Image.new("RGB", (1600, 900), color=(180, 120, 60))
    ref_emb_fp32 = engine_fp32.encode_image(ref_image)

    # ---- Dynamic quantization ----
    engine_dyn = EdgeMinerInference(
        checkpoint_path=checkpoint_path, index_dir=None, device="cpu", quantized=True,
    )
    engine_dyn.model = apply_dynamic_quantization(engine_dyn.model)
    results["int8_dynamic"] = engine_dyn.benchmark(
        num_images=num_timed, batch_size=batch_size
    )
    emb_dyn = engine_dyn.encode_image(ref_image)
    results["int8_dynamic"]["cosine_similarity_to_fp32"] = float(
        np.dot(ref_emb_fp32, emb_dyn) /
        (np.linalg.norm(ref_emb_fp32) * np.linalg.norm(emb_dyn))
    )

    # ---- Static quantization ----
    engine_static = EdgeMinerInference(
        checkpoint_path=checkpoint_path, index_dir=None, device="cpu", quantized=True,
    )
    # Calibration set: synthetic but varied. Real prod would use a dataset sample.
    calib_images = [
        Image.new("RGB", (1600, 900), color=tuple(np.random.randint(0, 255, 3).tolist()))
        for _ in range(32)
    ]
    engine_static.model = apply_static_quantization(
        engine_static.model, calib_images, engine_static
    )
    results["int8_static"] = engine_static.benchmark(
        num_images=num_timed, batch_size=batch_size
    )
    emb_static = engine_static.encode_image(ref_image)
    results["int8_static"]["cosine_similarity_to_fp32"] = float(
        np.dot(ref_emb_fp32, emb_static) /
        (np.linalg.norm(ref_emb_fp32) * np.linalg.norm(emb_static))
    )

    # ---- Speedups vs FP32 ----
    fp32_throughput = results["fp32"]["throughput_images_per_sec"]
    for mode in ["int8_dynamic", "int8_static"]:
        results[mode]["speedup_vs_fp32"] = round(
            results[mode]["throughput_images_per_sec"] / fp32_throughput, 2
        )

    return results


def print_benchmark_table(results: dict):
    """Pretty-print the benchmark dict as a markdown table."""
    print("\n| Mode | Throughput (img/s) | ms/img | Speedup | Cosine sim to FP32 |")
    print("|------|--------------------:|-------:|--------:|-------------------:|")
    for mode, m in results.items():
        speedup = f"{m.get('speedup_vs_fp32', 1.0):.2f}x" if "speedup_vs_fp32" in m else "1.00x"
        cosine = m.get("cosine_similarity_to_fp32", 1.0)
        print(f"| {mode} | {m['throughput_images_per_sec']:.2f} | "
              f"{m['ms_per_image']:.2f} | {speedup} | {cosine:.4f} |")
