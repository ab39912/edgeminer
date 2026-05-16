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