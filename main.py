from utils import process_images_as_video
from tracking import ObjectTracker, KeypointsTracker
from club_assignment import ClubAssigner, Club
from annotation import FootballVideoProcessor
from test_clips import test_clip_list
from player_info import CalculatePlayerStats
import os
import time
import numpy as np
import cv2

def main():
    """
    Main function to demonstrate how to use the football analysis project.
    This script will walk you through loading models, assigning clubs, tracking objects and players, and processing the video.
    """
    cv2.setUseOptimized(True)

    for clip in test_clip_list:
        start_time = time.time()

        image_dir = f'input_videos/test/{clip}/img1'
        output_dir = f'output_videos/test/{clip}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        else:
            print(f"Directory already exists: {output_dir}")

        # 1. Load the object detection model
        # Adjust the 'conf' value as per your requirements.
        obj_tracker = ObjectTracker(
            player_model_path='models/weights/player-model.pt',    # Object Detection Model Path for players
            ball_model_path='models/weights/ball-model.pt',     # Object Detection Model Path for ball
            conf=.5,                                            # Object Detection confidence threshold
            ball_conf=.05,                                        # Ball Detection confidence threshold
        )

        # 2. Load the keypoints detection model
        # Adjust the 'conf' and 'kp_conf' values as per your requirements.
        kp_tracker = KeypointsTracker(
            model_path='models/weights/keypoints-detection.pt', # Keypoints Model Weights Path
            conf=.5,                                            # Field Detection confidence threshold
            kp_conf=.5,                                         # Keypoint confidence threshold
            det_size=1280,                                       # Original size the model is trained on
        )
        
        # 3. Assign clubs to players based on their uniforms' colors
        # Create 'Club' objects - Needed for Player Club Assignment
        # Replace the RGB values with the actual colors of the clubs.
        club1 = Club('Club1',         # club name 
                    # (232, 247, 248), # player jersey color (putih)
                    # (200, 202, 231), # player jersey (putih tua)
                    (243, 248, 247), # player jersey color (putih)
                    # (31, 40, 37) # goalkeeper jersey color black
                    # (93, 133, 181) # goalkeeper jersey (biru)
                    (61, 100, 39) # goalkeeper jersey (hijau)
                    )
        club2 = Club('Club2',         # club name 
                    # (116, 0, 14), # player jersey color (merah)
                    # (72, 22, 36), # player jersey (merah tua)
                    (134, 48, 51), # player jersey (merah)
                    # (204, 203, 208)  # goalkeeper jersey color (abu-abu)
                    # (148, 76, 25) # goalkeeper (orange tua)
                    (240, 246, 44) # goalkeeper jersey (kuning)
                    )   

        # Create a ClubAssigner Object to automatically assign players and goalkeepers 
        # to their respective clubs based on jersey colors.
        club_assigner = ClubAssigner(club1, club2)

        # 4. Define the keypoints for a top-down view of the football field (from left to right and top to bottom)
        # These are used to transform the perspective of the field.
        top_down_keypoints = np.array([
            [0, 0], [0, 57], [0, 122], [0, 229], [0, 293], [0, 351],             # 0-5 (left goal line)
            [32, 122], [32, 229],                                                # 6-7 (left goal box corners)
            [64, 176],                                                           # 8 (left penalty dot)
            [96, 57], [96, 122], [96, 229], [96, 293],                           # 9-12 (left penalty box)
            [263, 0], [263, 122], [263, 229], [263, 351],                        # 13-16 (halfway line)
            [431, 57], [431, 122], [431, 229], [431, 293],                       # 17-20 (right penalty box)
            [463, 176],                                                          # 21 (right penalty dot)
            [495, 122], [495, 229],                                              # 22-23 (right goal box corners)
            [527, 0], [527, 57], [527, 122], [527, 229], [527, 293], [527, 351], # 24-29 (right goal line)
            [210, 176], [317, 176]                                               # 30-31 (center circle leftmost and rightmost points)
        ])

        # 5. Initialize the video processor
        # This processor will handle every task needed for analysis.
        processor = FootballVideoProcessor(obj_tracker,                                   # Created ObjectTracker object
                                        kp_tracker,                                    # Created KeypointsTracker object
                                        club_assigner,                                 # Created ClubAssigner object
                                        top_down_keypoints,                            # Created Top-Down keypoints numpy array
                                        field_img_path='input_videos/field_2d_v2.png', # Top-Down field image path
                                        save_tracks_dir=output_dir,               # Directory to save tracking information.
                                        draw_frame_num=True                            # Whether or not to draw current frame number on 
                                                                                        #the output video.
                                        )
        
        # 6. Process the video
        process_images_as_video(processor,
                                image_dir=image_dir,
                                output_video=f'{output_dir}/result.mp4',
                                batch_size=10,
                                fps=25,
                                name=clip,
                                )
        
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds")

        player_stat = CalculatePlayerStats(output_dir)
        player_stat.distance_stat()
        player_stat.speed_stat()

    os._exit(0)  # Force exit the program

if __name__ == '__main__':
    main()
