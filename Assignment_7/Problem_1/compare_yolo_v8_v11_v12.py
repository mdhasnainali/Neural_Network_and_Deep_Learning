import cv2
import time
from ultralytics import YOLO
import os

# === CONFIGURATION ===
video_path = "video.mp4"  

models = {
    "YOLOv8": YOLO("yolov8n.pt"),   
    "YOLOv11": YOLO("yolo11n.pt"), 
    "YOLOv12": YOLO("yolo12n.pt")  
}

# === PROCESSING FUNCTION ===
def evaluate_model(model_name, model, video_path, max_frames=1000):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_time = 0.0
    total_objects = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        results = model(frame, verbose=False)
        end = time.time()

        total_time += (end - start)
        boxes = results[0].boxes
        total_objects += len(boxes)

        frame_count += 1

    cap.release()

    avg_fps = frame_count / total_time
    avg_objects = total_objects / frame_count if frame_count else 0

    print(f"== {model_name} ==")
    print(f"Processed {frame_count} frames")
    print(f"Avg FPS: {avg_fps:.2f}")
    print(f"Avg Objects per Frame: {avg_objects:.2f}")
    print("-" * 30)

    return {
        "model": model_name,
        "fps": avg_fps,
        "objects_per_frame": avg_objects
    }

# === MAIN RUN ===
if __name__ == "__main__":
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        exit(1)

    results = []
    for name, model in models.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_model(name, model, video_path)
        results.append(metrics)

    print("=== Summary Comparison ===")
    for r in results:
        print(f"{r['model']}: FPS = {r['fps']:.2f}, Objects/Frame = {r['objects_per_frame']:.2f}")
