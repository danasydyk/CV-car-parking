import gradio as gr
import cv2
import numpy as np
import tempfile, os, time
from ultralytics import YOLO
from PIL import Image

# ── Load model ─────────────────────────────────────────────────────────────
model = YOLO("best.pt")
NAMES = model.names   # {0: "free", 1: "not_free"}

COLORS = {
    0: (74, 144, 217),    # blue  — free
    1: (231, 76,  60),    # red   — not_free
}

# ── Inference helpers ───────────────────────────────────────────────────────
def run_on_frame(frame_bgr, conf=0.25):
    results = model.predict(frame_bgr, conf=conf, verbose=False)[0]
    annotated = results.plot()
    boxes = results.boxes
    free     = int((boxes.cls == 0).sum())
    not_free = int((boxes.cls == 1).sum())
    return annotated, free, not_free

def predict_image(image: Image.Image, conf: float):
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    t0 = time.perf_counter()
    annotated, free, not_free = run_on_frame(frame, conf)
    ms = (time.perf_counter() - t0) * 1000

    out_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    summary = (
        f"🟦 Free spaces   : {free}\n"
        f"🟥 Occupied      : {not_free}\n"
        f"📦 Total detected: {free + not_free}\n"
        f"⏱️  Inference     : {ms:.1f} ms"
    )
    return Image.fromarray(out_rgb), summary

def predict_video(video_path: str, conf: float):
    cap = cv2.VideoCapture(video_path)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))

    frame_times, total_free, total_not_free, n_frames = [], 0, 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        annotated, free, not_free = run_on_frame(frame, conf)
        frame_times.append((time.perf_counter() - t0) * 1000)
        total_free += free
        total_not_free += not_free
        n_frames += 1
        writer.write(annotated)

    cap.release()
    writer.release()

    avg_ms  = np.mean(frame_times) if frame_times else 0
    avg_fps = 1000 / avg_ms if avg_ms > 0 else 0
    avg_free     = total_free     / max(n_frames, 1)
    avg_not_free = total_not_free / max(n_frames, 1)

    summary = (
        f"🎞️  Frames processed : {n_frames}\n"
        f"🟦 Avg free / frame  : {avg_free:.1f}\n"
        f"🟥 Avg occupied      : {avg_not_free:.1f}\n"
        f"⏱️  Avg inference     : {avg_ms:.1f} ms/frame\n"
        f"🚀 Effective FPS     : {avg_fps:.1f}"
    )
    return out_path, summary

# ── UI ─────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Parking Space Detector") as demo:
    gr.Markdown(
        """
        # 🅿️ Parking Space Detector
        Upload an **image** or **video** of a parking lot.
        The model detects **free** (blue) and **occupied** (red) spaces.
        """
    )

    conf_slider = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Confidence threshold")

    with gr.Tab("Image"):
        with gr.Row():
            img_input  = gr.Image(type="pil", label="Input image")
            img_output = gr.Image(type="pil", label="Annotated output")
        img_stats = gr.Textbox(label="Results", lines=4)
        gr.Button("Detect").click(
            predict_image,
            inputs=[img_input, conf_slider],
            outputs=[img_output, img_stats],
        )

    with gr.Tab("Video"):
        with gr.Row():
            vid_input  = gr.Video(label="Input video")
            vid_output = gr.Video(label="Annotated output")
        vid_stats = gr.Textbox(label="Results", lines=5)
        gr.Button("Detect").click(
            predict_video,
            inputs=[vid_input, conf_slider],
            outputs=[vid_output, vid_stats],
        )

    gr.Markdown(
        "Model: YOLOv8s fine-tuned on [Parking Space Detection Dataset](https://www.kaggle.com/datasets/trainingdatapro/parking-space-detection-dataset)"
    )

if __name__ == "__main__":
    demo.launch()
