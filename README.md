# edgeminer

## Training curves

![Recall@5 climbing from 6% to 97% over 16 epochs](notebooks/figures/val_recall@5.png)

InfoNCE loss converges smoothly, with held-out Recall@5 lifting from random-baseline (6.25%) to 97.5% by epoch 16.

## Results

Trained on **nuScenes mini** (10 scenes, 404 keyframes, 80/20 train/val split). All metrics measured on the 80-sample held-out set.

| Metric | Random baseline | EdgeMiner | Improvement |
|---|---|---|---|
| Recall@1 | 1.25% | **53.75%** | 43× |
| Recall@5 | 6.25% | **97.50%** | 16× |
| Recall@10 | 12.50% | **100.00%** | 8× |
| Val loss (InfoNCE) | 4.39 | **1.11** | — |

Training: 16 epochs, ~25 min on a single T4 GPU. Best checkpoint saved by val Recall@5.

> **Note on scale:** these metrics reflect performance on the 10-scene mini split, which contains some visually similar consecutive keyframes. Full nuScenes (1000 scenes) would present harder retrieval distractors. Results here demonstrate that the dual encoder learns a meaningful joint embedding space; scaling to full nuScenes is part of the project roadmap.


## Retrieval Demo

EdgeMiner supports four types of similarity search:

| Query type | Use case |
|---|---|
| Image → Image | Find scenes that look like a target scene |
| LiDAR → LiDAR | Find scenes with similar 3D structure |
| Image → LiDAR | Cross-modal: retrieve LiDAR scans matching a camera query |
| LiDAR → Image | Cross-modal: retrieve images matching a point cloud query |

![Top-5 visually similar scenes retrieved by EdgeMiner](notebooks/figures/query_type.png)

*Top-5 visually similar scenes for a sample query. The trained dual encoder retrieves adjacent keyframes from the same driving sequence (similarity ~0.6+) followed by visually related scenes from other parts of the dataset.*


### Night Scene Mining

![Night driving scenes retrieved from a single query](notebooks/figures/lidar_1.png)

**Query:** Scene 317 — *"Night, big street, bus stop, high speed, construction vehicle"*

EdgeMiner's top-5 retrievals are all night urban driving scenes with consistent lighting, road type, and composition. The dual encoder learned night-time as a distinct semantic mode without explicit lighting labels — purely from contrastive pretraining on paired camera-LiDAR data.

This is the foundational workflow for edge-case mining at scale: given one example of a rare scenario (e.g., construction vehicle present at night), retrieve all similar instances from the dataset for labeling and retraining.

### Supported Query Types

| Query type | Use case |
|---|---|
| Image → Image | Find scenes that look visually similar |
| LiDAR → LiDAR | Find scenes with similar 3D structure |
| Image → LiDAR | Cross-modal: retrieve LiDAR scans from a camera query |
| LiDAR → Image | Cross-modal: retrieve images from a point cloud query |

## Active Learning: The Failure Atlas

EdgeMiner closes the active learning loop: a pretrained YOLOv8-large detector is run over every scene, per-scene uncertainty is scored via a composite of mean/min confidence, low-confidence ratio, class entropy, and missed-detection rate. The top-10% most uncertain scenes are flagged for labeling.

![UMAP projection of all scenes colored by detector uncertainty](notebooks/figures/failure_atlas.png)

*UMAP-projected embeddings of all 404 nuScenes mini scenes. Color = composite uncertainty score. Red dots are the top-10% most uncertain scenes — they cluster systematically rather than scattering, indicating structured failure modes rather than random model noise.*

### Mined Failure Cluster

![Night intersection failure cluster](notebooks/figures/cluster_1.png)

A representative failure cluster mined automatically by the pipeline. **Anchor** (left): a night intersection scene with lens flare; the detector produces 24 boxes but at low confidence (composite uncertainty = 0.73). **Cluster members**: four visually near-identical scenes retrieved via FAISS (similarity > 0.96), representing the "lens-flare-at-night" failure mode. This is one labeling batch — find one failure, automatically surface the similar failures, label them together for the next training round.

> **Why this matters:** Identifying long-tail failure modes is the single hardest problem in autonomous driving development. Production systems like Motional's Omnitag tackle this at petabyte scale. EdgeMiner demonstrates the same workflow end-to-end on a 10-scene dataset, validating the methodology before scaling.
