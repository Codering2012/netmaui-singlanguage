import cv2
import numpy as np
import os

def create_sample_video(filename, text, duration_sec=5, fps=20, width=640, height=480):
    print(f"Generating {filename}...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    # Define colors
    bg_color = (31, 24, 17) # Dark background
    text_color = (255, 255, 255)
    accent_color = (246, 130, 59) # Blue/Orange accent

    for frame_num in range(duration_sec * fps):
        # Create background
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        # Add a moving circle to make it look like a real video
        center_x = int(width / 2 + 100 * np.sin(frame_num / 5))
        center_y = int(height / 2 + 50 * np.cos(frame_num / 5))
        cv2.circle(frame, (center_x, center_y), 40, accent_color, -1)

        # Add text
        cv2.putText(frame, "SIGN LANGUAGE LESSON", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, accent_color, 2)
        cv2.putText(frame, text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
        cv2.putText(frame, f"Frame: {frame_num}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

        # Progress bar
        progress = frame_num / (duration_sec * fps)
        cv2.rectangle(frame, (50, 430), (50 + int((width - 100) * progress), 440), accent_color, -1)
        cv2.rectangle(frame, (50, 430), (width - 50, 440), (100, 100, 100), 1)

        out.write(frame)

    out.release()
    print(f"Successfully created {filename}")

if __name__ == "__main__":
    output_dir = "VIDEO"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    create_sample_video(os.path.join(output_dir, "lesson_alphabet_basics.mp4"), "Alphabet Basics: A, B, C")
    create_sample_video(os.path.join(output_dir, "lesson_common_phrases.mp4"), "Common Phrases: Hello, Thanks")
