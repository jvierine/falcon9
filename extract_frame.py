import cv2
import sys
import os
import numpy as np
import numpy as n

def save_first_25_average(video_path):
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    frames = []
    for i in range(50):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.astype(np.float32))  # store as float for averaging

    cap.release()

    if not frames:
        print("Error: No frames could be read.")
        return

    # Average frames
    avg_frame = np.mean(frames, axis=0).astype(np.uint8)
    # high pass filter image
    kernel = n.zeros([50,50])
    kernel[:,:]=1.0/(50*50)#100.0
    low=n.array(cv2.filter2D(avg_frame,-1,kernel),dtype=n.float32)
    avg_frame=n.array(avg_frame,dtype=n.float32)-low
    avg_frame=255.0*(avg_frame-n.min(avg_frame))/(n.max(avg_frame)-n.min(avg_frame))
    avg_frame=n.array(avg_frame,dtype=n.uint8)


    # Create output filename
    base = os.path.splitext(video_path)[0]
    output_image = f"{base}_avg_first25.png"

    cv2.imwrite(output_image, avg_frame)
    print(f"Saved average of first {len(frames)} frames as {output_image}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python save_first_frame.py <video_file>")
        sys.exit(1)

    video_file = sys.argv[1]
    save_first_25_average(video_file)
