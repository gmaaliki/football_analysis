import cv2
import os
import glob
import tempfile
import traceback
from typing import Tuple, Optional
from gta_link.generate_tracklets import generate_tracklets
from gta_link.refine_tracklets import refine_tracklets
from tqdm import tqdm
import multiprocessing

def _convert_frames_to_video(frame_dir: str, output_video: str, fps: float, frame_size: Tuple[int, int]) -> None:
    """
    Convert frames in a directory to a video file.

    Args:
        frame_dir (str): Directory containing frame images.
        output_video (str): Path to save the output video.
        fps (float): Frames per second for the output video.
        frame_size (Tuple[int, int]): Size of the frames as (width, height).
    """
    if os.path.exists(output_video):
        os.remove(output_video)
        print(f"{output_video} has been deleted.")
    else:
        print(f"{output_video} does not exist.")

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)
    
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    frame_count = len(frame_files)

    if frame_count <= 0:
        out.release()
        print("There are no frames to save")
        return
    
    for filename in frame_files:
        img = cv2.imread(filename)
        out.write(img)
    
    out.release()
    print(f"Video saved as {output_video}")

def process_images_as_video(processor = None, image_dir: str = 0, output_video: Optional[str] = "output.mp4", 
                  batch_size: int = 30, skip_seconds: int = 0, fps: int = 30, name: str = "") -> None:
    """
    Process a video file or stream, capturing, processing, and displaying frames.

    Args:
        processor (AbstractVideoProcessor): Object responsible for processing frames.
        video_source (str, optional): Video source (default is "0" for webcam).
        output_video (Optional[str], optional): Path to save the output video or None to skip saving.
        batch_size (int, optional): Number of frames to process at once.
        skip_seconds (int, optional): Seconds to skip at the beginning of the video.
    """
    from annotation import AbstractVideoProcessor  # Lazy import

    if processor is not None and not isinstance(processor, AbstractVideoProcessor):
        raise ValueError("The processor must be an instance of AbstractVideoProcessor.")
    
    if not image_dir or not os.path.exists(image_dir):
        print("Error: Image directory is not valid")
        return

    if len(os.listdir(image_dir)) == 0:
        print("Error: No images present")
        return

    images = [os.path.join(image_dir, frame) for frame in os.listdir(image_dir)]
    total_frame = len(images)

    print(f"Video FPS: {fps}")
    print(f"Total frame count: {total_frame}")
    frames_to_skip = int(skip_seconds * fps)

    # Skip the first 'frames_to_skip' frames
    images = images[frames_to_skip:]
    frame_count = frames_to_skip  # Start counting frames from here
    frames = []

    try:
        while frame_count < total_frame:
            image_name = images[frame_count]
            frame = cv2.imread(image_name)
            resized_frame = cv2.resize(frame, (1920, 1080))

            frames.append(resized_frame)
            frame_count += 1
    except Exception as e:
        print(f"Error in frame capture: {e}")
    print("Frame capture complete")

    print("Starting frame processing")
    try:
        for i in tqdm(range(0, len(frames), batch_size), desc="Processing batches"):
            frames_to_process = frames[i:i+batch_size]
            processor.process(frames_to_process, fps)
    except Exception as e:
        print(f"Error in frame processing: {e}")

    print("Frame processing complete")

    processor.export_mot(name=name)
    print("Results converted to MOT format")

    # Refine tracking results with Global Tracklet Association
    generate_tracklets(
        model_path = "./gta_link/reid_checkpoints\sports_model.pth.tar-60",
        data_path = f"./input_videos/test/{name}",
        pred_file = f"./output_videos/mot_results/{name}.txt",
        output_dir = f"./output_videos/pickle",
    )

    refined_mot_path = refine_tracklets(
        track_src = f"./output_videos/pickle/{name}.pkl",
        output_dir = "./output_videos",

        use_connect = True,
        use_split = True,
        min_len = 100, # default 100
        eps = 0.6, # default 0.6
        min_samples = 10, # default 10
        max_k = 3, # default 3
        spatial_factor = 1.0, # default 1.0
        merge_dist_thres = 0.4, # default 0.4
    )


    kobj_file_path = f"./output_videos/test/{name}/keypoint_tracks.json"
    processed_frames = processor.process_from_mot_export_and_kobj_json(frames=frames, fps=fps, mot_file_path=refined_mot_path, kobj_file_path=kobj_file_path)

    processor.extract_speed()
    width = 1920
    height = 1080

    with tempfile.TemporaryDirectory() as temp_dir:
        print("Starting convert to video")
        try:
            for i in processed_frames:
                if i is None:
                    print("No more frames to display")
                    break
                frame_count, processed_frame = i

                frame_filename = os.path.join(temp_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, processed_frame)
        except Exception as e:
            print(f"Error displaying frame: {e}")

        try:
            if output_video is not None:
                print("Converting frames to video...")
                _convert_frames_to_video(temp_dir, output_video, fps, (width, height))

        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()

    print("Video processing completed. Program will now exit.")