# 🅿️ Parking Space Detection

A YOLOv8-based model for detecting free and occupied parking spaces from overhead images and video.
https://github.com/user-attachments/assets/02566c4f-1faf-43fb-9e62-cde0100cf81d

## Demo

Try it live on Hugging Face Spaces: [coolbambook/car_parking](https://huggingface.co/spaces/coolbambook/car_parking)

## Dataset

Training data: [Parking Space Detection Dataset](https://www.kaggle.com/datasets/trainingdatapro/parking-space-detection-dataset) (TrainingDataPro on Kaggle)

- 30 images, 903 annotated parking spaces
- Original labels: `free_parking_space`, `not_free_parking_space`, `partially_free_parking_space`
- `partially_free` (6 samples) was merged into `not_free` due to insufficient examples
- Final classes: **free** (273 boxes) | **not_free** (630 boxes)
- 80/20 train/val split, seed 42

## Model

- Architecture: YOLOv8s (pretrained on COCO, fine-tuned)
- Epochs: 50
- Image size: 640
- Augmentation: horizontal and vertical flips only
- Class weight: `cls_pw=0.43` to compensate for class imbalance

## Results

| Metric | Value |
|---|---|
| mAP@50 | 0.943 |
| mAP@50-95 | 0.840 |
| Precision | 0.897 |
| Recall | 0.991 |

**Per-class:**

| Class | Precision | Recall | AP@50 |
|---|---|---|---|
| free | 0.861 | 0.992 | 0.925 |
| not_free | 0.933 | 0.990 | 0.961 |

**Inference speed:** ~53 FPS on NVIDIA T4 GPU (18ms mean latency).

## Usage

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("parking_lot.jpg", conf=0.25)
results[0].show()
```

## Run Locally

```bash
git clone https://github.com/your-username/parking-space-detection
cd parking-space-detection
pip install -r requirements.txt
python app.py
```

## Known Limitations

The model may predict free spaces on open road sections with no vehicles, as it learned to distinguish occupied vs. empty areas rather than explicitly learning parking slot boundaries. This is a consequence of the small training set (30 images) — the model never saw road areas as negative examples, so it generalizes "no car = free space."

This would improve with:
- More training images covering diverse lot layouts and camera angles
- Hard negative examples (road areas with no parking boxes)
- ROI masking for fixed-camera deployments

Contributions and dataset suggestions are welcome.

## Project Structure

```
├── app.py                  # Gradio app (image + video inference)
├── best.pt                 # Trained model weights
├── requirements.txt
├── prepare_and_train.py    # XML → YOLO conversion + training script
├── explore_parking.py      # Dataset exploration
└── benchmark_fps.py        # Inference speed benchmark
```

