import cv2
import os

def merge_letters(word, output_filename, letter_dir="ASL_LETTERS"):
    word = word.upper()
    valid_letters = [l for l in word if os.path.exists(os.path.join(letter_dir, f"{l}.mp4"))]
    
    if not valid_letters:
        print(f"No valid letter videos found for word: {word}")
        return

    # Initialize video writer with settings from the first video
    first_video_path = os.path.join(letter_dir, f"{valid_letters[0]}.mp4")
    cap = cv2.VideoCapture(first_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Use 'mp4v' or 'XVID' codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    print(f"Generating {output_filename} for word: {word}")

    for letter in valid_letters:
        letter_path = os.path.join(letter_dir, f"{letter}.mp4")
        cap = cv2.VideoCapture(letter_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Ensure frame size matches
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            
            out.write(frame)
        
        cap.release()
        # Add a few freeze frames at the end of each letter for clarity (optional)
        # for _ in range(5):
        #     out.write(frame)

    out.release()
    print(f"Successfully created {output_filename}")

if __name__ == "__main__":
    # Generate the requested alphabet sequences
    merge_letters("ABCDEFGHIJKLM", "VIDEO/Letters_Alphabet A-M.mp4")
    merge_letters("NOPQRSTUVWXYZ", "VIDEO/Letters_Alphabet N-Z.mp4")
    
    # Generate a dynamic gesture example word
    merge_letters("ASL", "VIDEO/Letters_Dynamic Gestures.mp4")
