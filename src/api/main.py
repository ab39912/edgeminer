"""
FastAPI service for EdgeMiner.

Endpoints:
    GET  /health           - liveness probe
    POST /embed            - upload an image, get its 256-d embedding
    POST /search           - upload an image, get top-K similar scenes
    GET  /benchmark        - run a throughput benchmark
    GET  /docs             - auto-generated Swagger UI (free from FastAPI)

Run locally:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Configuration via environment variables:
    EDGEMINER_CHECKPOINT  - path to trained .pt checkpoint
    EDGEMINER_INDEX_DIR   - path to FAISS index directory
    EDGEMINER_QUANTIZED   - "1" to load INT8 model, else FP32
    EDGEMINER_DEVICE      - "cpu" or "cuda"
"""

import os
import time
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from src.api.inference import EdgeMinerInference
    from src.api.quantization import apply_dynamic_quantization
except ImportError:
    pass


# ---------- Config from environment ----------

CHECKPOINT_PATH = os.environ.get(
    "EDGEMINER_CHECKPOINT", "checkpoints/best.pt"
)
INDEX_DIR = os.environ.get("EDGEMINER_INDEX_DIR", "index")
QUANTIZED = os.environ.get("EDGEMINER_QUANTIZED", "0") == "1"
DEVICE = os.environ.get("EDGEMINER_DEVICE", "cpu")


# ---------- Pydantic schemas (FastAPI generates docs from these) ----------

class EmbeddingResponse(BaseModel):
    embedding: List[float] = Field(..., description="256-d L2-normalized vector")
    dim: int = Field(..., description="Embedding dimensionality")
    inference_ms: float = Field(..., description="Server-side inference latency in ms")


class SearchResult(BaseModel):
    rank: int
    sample_token: str
    similarity: float
    index: int


class SearchResponse(BaseModel):
    query_inference_ms: float
    results: List[SearchResult]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    index_loaded: bool
    quantized: bool
    device: str
    num_scenes: Optional[int]


# ---------- App init ----------

app = FastAPI(
    title="EdgeMiner Inference API",
    description=(
        "Multimodal data mining service for autonomous driving. "
        "Upload an image, get back the 256-d embedding or the top-K most "
        "similar scenes from the indexed dataset."
    ),
    version="0.1.0",
)

# CORS so a Streamlit frontend (or anything else) can call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Lazy-load the inference engine at startup
@app.on_event("startup")
def load_engine():
    global engine
    try:
        engine = EdgeMinerInference(
            checkpoint_path=CHECKPOINT_PATH,
            index_dir=INDEX_DIR,
            device=DEVICE,
            quantized=QUANTIZED,
        )
        if QUANTIZED:
            engine.model = apply_dynamic_quantization(engine.model)
            print("Applied dynamic INT8 quantization")
    except Exception as e:
        print(f"WARNING: Could not load engine at startup: {e}")
        engine = None


# ---------- Routes ----------

@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness + readiness probe. Use this in Docker healthchecks."""
    return HealthResponse(
        status="ok" if engine is not None else "degraded",
        model_loaded=engine is not None,
        index_loaded=engine is not None and engine.index is not None,
        quantized=QUANTIZED,
        device=DEVICE,
        num_scenes=(len(engine.index.sample_tokens) if engine and engine.index else None),
    )


@app.post("/embed", response_model=EmbeddingResponse)
async def embed(file: UploadFile = File(..., description="Image to embed (JPEG or PNG)")):
    """Upload an image and return its 256-d embedding."""
    if engine is None:
        raise HTTPException(503, "Inference engine not loaded")
    contents = await file.read()
    start = time.perf_counter()
    embedding = engine.encode_image(contents)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return EmbeddingResponse(
        embedding=embedding.tolist(),
        dim=embedding.shape[0],
        inference_ms=round(elapsed_ms, 2),
    )


@app.post("/search", response_model=SearchResponse)
async def search(
    file: UploadFile = File(..., description="Query image"),
    k: int = Query(5, ge=1, le=50, description="Number of results"),
    target_modality: str = Query("image", regex="^(image|lidar)$"),
):
    """
    Upload an image and retrieve the top-K most similar scenes from the indexed dataset.

    target_modality=image: find visually similar scenes
    target_modality=lidar: cross-modal retrieval (image -> matching LiDAR scenes)
    """
    if engine is None:
        raise HTTPException(503, "Inference engine not loaded")
    if engine.index is None:
        raise HTTPException(503, "FAISS index not loaded")

    contents = await file.read()
    start = time.perf_counter()
    embedding = engine.encode_image(contents)
    results = engine.search(embedding, k=k, target_modality=target_modality)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return SearchResponse(
        query_inference_ms=round(elapsed_ms, 2),
        results=[
            SearchResult(rank=i + 1, **r) for i, r in enumerate(results)
        ],
    )


@app.get("/benchmark")
def benchmark(
    num_images: int = Query(50, ge=10, le=500),
    batch_size: int = Query(16, ge=1, le=64),
):
    """Run an in-process throughput benchmark on synthetic data."""
    if engine is None:
        raise HTTPException(503, "Inference engine not loaded")
    return engine.benchmark(num_images=num_images, batch_size=batch_size)
