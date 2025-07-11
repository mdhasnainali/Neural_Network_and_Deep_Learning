import cv2
import os

# === CONFIGURATION ===
video_path = "video.mp4"  # Change this
output_folder = "video_chunks"
chunk_duration_minutes = 1

os.makedirs(output_folder, exist_ok=True)

# === MAIN SPLITTING FUNCTION ===
def split_video(video_path, output_folder, chunk_duration_minutes):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    chunk_size = int(fps * 60 * chunk_duration_minutes)  # frames per chunk

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    chunk_index = 0
    frame_index = 0
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % chunk_size == 0:
            if out:
                out.release()
            chunk_filename = os.path.join(output_folder, f"chunk_{chunk_index:03}.mp4")
            out = cv2.VideoWriter(chunk_filename, fourcc, fps, (width, height))
            print(f"Started writing {chunk_filename}")
            chunk_index += 1

        out.write(frame)
        frame_index += 1

    if out:
        out.release()
    cap.release()
    print("✅ Video splitting complete.")

# === RUN ===
if __name__ == "__main__":
    split_video(video_path, output_folder, chunk_duration_minutes)
