import cv2
import os
from ultralytics import YOLO

video_path = "video_2.mp4"          
output_dir = "output_videos"
save_frames_dir = "saved_frames" 
frame_save_interval = 100        

os.makedirs(output_dir, exist_ok=True)
os.makedirs(save_frames_dir, exist_ok=True)  

models = {
    "YOLOv8": YOLO("yolov8n.pt"),
    "YOLOv11": YOLO("yolo11n.pt"), 
    "YOLOv12": YOLO("yolo12n.pt")  
}

# ======= FUNCTION TO PROCESS VIDEO =======
def process_video_with_model(model_name, model, video_path, output_path, frames_output_dir, save_interval):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing with {model_name}...")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model(frame, verbose=False) 

        annotated_frame = results[0].plot()

        if frame_count % save_interval == 0:
            frame_filename = os.path.join(frames_output_dir, f"{model_name}_frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, annotated_frame)
            print(f"Saved frame {frame_count} for {model_name} at {frame_filename}")

        cv2.imshow(f"{model_name} Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"User pressed 'q', stopping processing for {model_name}.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows() #\
    print(f"Finished processing with {model_name}.")
    print(f"Saved output video for {model_name} at {output_path}")
    print(f"Saved selected frames for {model_name} in {frames_output_dir}")

# ======= MAIN LOOP =======
if __name__ == "__main__":
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at: {video_path}")
        print("Please ensure 'video_2.mp4' is in the same directory as the script, or provide the full path.")
        exit(1)

    for name, model in models.items():
        output_file = os.path.join(output_dir, f"{name}_output.mp4")
        process_video_with_model(name, model, video_path, output_file, save_frames_dir, frame_save_interval)

