"""
Streamlit frontend for EdgeMiner.

A minimal UI for the FastAPI service. Upload an image, view its embedding,
or search for similar driving scenes.

Run locally:
    streamlit run src/frontend/app.py

Deploy to Hugging Face Spaces:
    1. Push this file and requirements.txt to a Space
    2. Set EDGEMINER_API_URL to your deployed FastAPI URL
"""

import os
import io

import requests
import streamlit as st
from PIL import Image


# ---------- Config ----------

API_URL = os.environ.get("EDGEMINER_API_URL", "http://localhost:8000")


# ---------- Page setup ----------

st.set_page_config(
    page_title="EdgeMiner",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 EdgeMiner")
st.markdown(
    "Multimodal data mining for autonomous driving. "
    "Upload a driving scene image to find visually similar scenes from the indexed dataset."
)


# ---------- Sidebar: API status ----------

with st.sidebar:
    st.header("API Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        st.success("Connected" if health.get("status") == "ok" else "Degraded")
        st.json(health)
    except Exception as e:
        st.error(f"Cannot reach API at {API_URL}")
        st.caption(f"Error: {e}")
        st.stop()

    st.divider()
    st.header("Settings")
    k = st.slider("Number of results", min_value=1, max_value=10, value=5)
    modality = st.radio(
        "Target modality",
        options=["image", "lidar"],
        help="image = find visually similar scenes; lidar = cross-modal retrieval",
    )


# ---------- Main: upload + results ----------

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload a query image")
    uploaded = st.file_uploader(
        "Driving scene (JPEG or PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is not None:
        query_image = Image.open(uploaded).convert("RGB")
        st.image(query_image, caption="Query", use_container_width=True)

with col2:
    if uploaded is not None:
        st.subheader(f"Top {k} similar scenes")

        # Re-serialize so we can send to the API
        buf = io.BytesIO()
        query_image.save(buf, format="JPEG")
        buf.seek(0)

        with st.spinner("Searching..."):
            try:
                response = requests.post(
                    f"{API_URL}/search",
                    files={"file": ("query.jpg", buf, "image/jpeg")},
                    params={"k": k, "target_modality": modality},
                    timeout=15,
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.stop()

        st.caption(f"Latency: {data['query_inference_ms']} ms")

        # Render results as a horizontal strip
        cols = st.columns(min(k, 5))
        for i, result in enumerate(data["results"]):
            with cols[i % 5]:
                st.metric(
                    label=f"#{result['rank']}",
                    value=f"sim {result['similarity']:.3f}",
                )
                st.caption(f"`{result['sample_token'][:8]}...`")

        # Show full result data for transparency
        with st.expander("Raw response"):
            st.json(data)
    else:
        st.info("← Upload a driving scene image to start.")


# ---------- Footer ----------

st.divider()
st.caption(
    "EdgeMiner runs a dual encoder (DINOv2 + PointNet) trained with contrastive "
    "learning on nuScenes, indexed with FAISS for fast retrieval. "
    "[Source on GitHub](https://github.com/your-username/edgeminer)"
)
