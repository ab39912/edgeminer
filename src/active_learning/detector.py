"""
YOLOv8-large detector wrapper for EdgeMiner active learning.

Runs the pretrained model over all nuScenes camera images and saves
detections (boxes, classes, confidences) for downstream uncertainty scoring.

We use COCO-pretrained YOLOv8l because:
  - Strong baseline performance (failures are meaningful)
  - Detects the classes we care about (person, car, truck, bus, bicycle, etc.)
  - Pretrained weights are public and reproducible
  - Single-model inference (no ensemble needed for v1)

Usage in Colab:
    from src.active_learning.detector import run_detector_over_dataset
    detections = run_detector_over_dataset(
        dataroot='/content/data/nuscenes',
        output_path='/content/drive/MyDrive/edgeminer/detections/yolov8l.pt',
    )
"""

from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from PIL import Image

try:
    from nuscenes.nuscenes import NuScenes
except ImportError:
    pass


# COCO classes we care about for driving scenes. YOLOv8 outputs COCO class IDs.
# Index is the COCO class id, value is the human-readable label.
DRIVING_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic_light",
    11: "stop_sign",
}


def load_yolo_model(model_size: str = "yolov8l.pt"):
    """Load a pretrained YOLOv8 model via ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. In Colab: !pip install ultralytics -q")

    model = YOLO(model_size)
    return model


@torch.no_grad()
def run_detector_over_dataset(
    dataroot: str,
    output_path: str,
    version: str = "v1.0-mini",
    camera_channel: str = "CAM_FRONT",
    model_size: str = "yolov8l.pt",
    confidence_threshold: float = 0.05,  # low to capture uncertain detections too
    device: Optional[str] = None,
) -> dict:
    """
    Run YOLOv8 over every front-camera image in the dataset.

    Returns a dict mapping sample_token -> detection summary:
        {
            sample_token: {
                "num_detections": int,
                "boxes": (N, 4) tensor of xyxy,
                "classes": (N,) tensor of class ids,
                "confidences": (N,) tensor of [0, 1] scores,
                "class_names": list of N human-readable labels,
            },
            ...
        }
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running detector on: {device}")

    model = load_yolo_model(model_size)
    print(f"Loaded {model_size}")

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    print(f"Processing {len(nusc.sample)} samples...")

    detections = {}
    for sample in tqdm(nusc.sample, desc="Detecting"):
        token = sample["token"]
        cam_data = nusc.get("sample_data", sample["data"][camera_channel])
        img_path = Path(dataroot) / cam_data["filename"]

        # YOLOv8 accepts file path, PIL Image, numpy, or tensor.
        # Pass file path so YOLO handles resizing internally.
        results = model.predict(
            source=str(img_path),
            conf=confidence_threshold,
            verbose=False,
            device=device,
        )
        result = results[0]  # single image

        boxes = result.boxes.xyxy.cpu()                    # (N, 4)
        classes = result.boxes.cls.cpu().int()              # (N,)
        confidences = result.boxes.conf.cpu()               # (N,)

        # Map class ids to names, keeping only driving-relevant classes
        class_names = []
        keep_mask = torch.zeros(len(classes), dtype=torch.bool)
        for i, c in enumerate(classes.tolist()):
            if c in DRIVING_CLASSES:
                class_names.append(DRIVING_CLASSES[c])
                keep_mask[i] = True

        detections[token] = {
            "num_detections": int(keep_mask.sum().item()),
            "boxes": boxes[keep_mask],
            "classes": classes[keep_mask],
            "confidences": confidences[keep_mask],
            "class_names": class_names,
            "num_total_detections": int(len(classes)),  # incl. non-driving classes
        }

    # Quick summary stats
    total_dets = sum(d["num_detections"] for d in detections.values())
    avg_conf = sum(d["confidences"].mean().item() if d["num_detections"] > 0 else 0
                   for d in detections.values()) / len(detections)
    print(f"\nTotal driving-class detections: {total_dets}")
    print(f"Average confidence per scene: {avg_conf:.3f}")
    print(f"Scenes with zero detections: "
          f"{sum(1 for d in detections.values() if d['num_detections'] == 0)}")

    # Save to disk
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "detections": detections,
        "model": model_size,
        "confidence_threshold": confidence_threshold,
        "driving_classes": DRIVING_CLASSES,
    }, output_path)
    print(f"Saved to: {output_path}")

    return detections


if __name__ == "__main__":
    run_detector_over_dataset(
        dataroot="/content/data/nuscenes",
        output_path="/content/drive/MyDrive/edgeminer/detections/yolov8l.pt",
    )
