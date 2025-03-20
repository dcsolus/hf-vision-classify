import cv2
from pathlib import Path

# Function to extract frames from a video
def extract_frames(video_file_path:Path, output_dir:Path):
    cap = cv2.VideoCapture(video_file_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # print(fps)
    frame_count = 0
    success, image = cap.read()
    while success:
        if frame_count % fps ==0: # every 1 second
            frame_time = frame_count // fps
            frame_filename = output_dir / f"frame_{frame_time}.jpg" 
            cv2.imwrite(frame_filename, image)  
        success, image = cap.read()     
        frame_count +=1

if __name__=='__main__':
    video_file = Path('video.mp4')
    output_dir = Path('frames')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extract_frames(video_file, output_dir)