from utils import process_video, process_images_as_video
from tracking import ObjectTracker, KeypointsTracker
from club_assignment import ClubAssigner, Club
from ball_to_player_assignment import BallToPlayerAssigner
from annotation import FootballVideoProcessor
from test_clips import test_clip_list
import os

import numpy as np

def main():
    """
    Main function to demonstrate how to use the football analysis project.
    This script will walk you through loading models, assigning clubs, tracking objects and players, and processing the video.
    """

    for clip in test_clip_list:

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
            conf=.3,                                            # Field Detection confidence threshold
            kp_conf=.5,                                         # Keypoint confidence threshold
            det_size=640,                                       # Original size the model is trained on
        )
        
        # 3. Assign clubs to players based on their uniforms' colors
        # Create 'Club' objects - Needed for Player Club Assignment
        # Replace the RGB values with the actual colors of the clubs.
        club1 = Club('Club1',         # club name 
                    (232, 247, 248), # player jersey color
                    (6, 25, 21)      # goalkeeper jersey color
                    )
        club2 = Club('Club2',         # club name 
                    (172, 251, 145), # player jersey color
                    (239, 156, 132)  # goalkeeper jersey color
                    )   

        # Create a ClubAssigner Object to automatically assign players and goalkeepers 
        # to their respective clubs based on jersey colors.
        club_assigner = ClubAssigner(club1, club2)

        # 4. Initialize the BallToPlayerAssigner object
        ball_player_assigner = BallToPlayerAssigner(club1, club2, fps=25)

        # 5. Define the keypoints for a top-down view of the football field (from left to right and top to bottom)
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

        # 6. Initialize the video processor
        # This processor will handle every task needed for analysis.
        processor = FootballVideoProcessor(obj_tracker,                                   # Created ObjectTracker object
                                        kp_tracker,                                    # Created KeypointsTracker object
                                        club_assigner,                                 # Created ClubAssigner object
                                        ball_player_assigner,                          # Created BallToPlayerAssigner object
                                        top_down_keypoints,                            # Created Top-Down keypoints numpy array
                                        field_img_path='input_videos/field_2d_v2.png', # Top-Down field image path
                                        save_tracks_dir=output_dir,               # Directory to save tracking information.
                                        draw_frame_num=True                            # Whether or not to draw current frame number on 
                                                                                        #the output video.
                                        )
        
        # 7. Process the video
        # Specify the input video path and the output video path. 
        # The batch_size determines how many frames are processed in one go.
        # process_video(processor,                                # Created FootballVideoProcessor object
        #               video_source='input_videos/video.mp4', # Video source (in this case video file path)
        #               output_video='output_videos/result.mp4',    # Output video path (Optional)
        #               batch_size=10                             # Number of frames to process at once
        #               )

        # 7.1 Process images as video
        # Processing multuple image sequence as video
        # Ensure the image file name is alphabetically ordered according to the video sequence
        # process_images_as_video(processor,
        #                         image_dir='input_videos/images',
        #                         output_video='output_videos/result.mp4',
        #                         batch_size=10,
        #                         fps=25,
        #                         )

        process_images_as_video(processor,
                                image_dir=image_dir,
                                output_video=f'{output_dir}/result.mp4',
                                batch_size=10,
                                fps=25,
                                name=clip,
                                )


    os._exit(0)  # Force exit the program

if __name__ == '__main__':
    main()
