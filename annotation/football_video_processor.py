from .abstract_annotator import AbstractAnnotator
from .abstract_video_processor import AbstractVideoProcessor
from .object_annotator import ObjectAnnotator
from .keypoints_annotator import KeypointsAnnotator
from .projection_annotator import ProjectionAnnotator
from position_mappers import ObjectPositionMapper
from speed_estimation import SpeedEstimator
from .frame_number_annotator import FrameNumberAnnotator
from file_writing import TracksJsonWriter
from tracking import ObjectTracker, KeypointsTracker
from club_assignment import ClubAssigner

import cv2
import numpy as np
import os
import json
from typing import List, Dict, Optional, Tuple

class FootballVideoProcessor(AbstractAnnotator, AbstractVideoProcessor):
    """
    A video processor for football footage that tracks objects and keypoints,
    estimates speed and adds various annotations.
    """

    def __init__(self, obj_tracker: ObjectTracker, kp_tracker: KeypointsTracker, 
                 club_assigner: ClubAssigner, top_down_keypoints: np.ndarray, field_img_path: str, 
                 save_tracks_dir: Optional[str] = None, draw_frame_num: bool = True) -> None:
        """
        Initializes the video processor with necessary components for tracking, annotations, and saving tracks.

        Args:
            obj_tracker (ObjectTracker): The object tracker for tracking players and balls.
            kp_tracker (KeypointsTracker): The keypoints tracker for detecting and tracking keypoints.
            club_assigner (ClubAssigner): Assigner to determine clubs for the tracked players.
            ball_to_player_assigner (BallToPlayerAssigner): Assigns the ball to a specific player based on tracking.
            top_down_keypoints (np.ndarray): Keypoints to map objects to top-down positions.
            field_img_path (str): Path to the image of the football field used for projection.
            save_tracks_dir (Optional[str]): Directory to save tracking information. If None, no tracks will be saved.
            draw_frame_num (bool): Whether or not to draw current frame number on the output video.
        """

        self.obj_tracker = obj_tracker
        self.obj_annotator = ObjectAnnotator()
        self.kp_tracker = kp_tracker
        self.kp_annotator = KeypointsAnnotator()
        self.club_assigner = club_assigner
        self.projection_annotator = ProjectionAnnotator()
        self.obj_mapper = ObjectPositionMapper(top_down_keypoints, alpha = 0.3)
        self.draw_frame_num = draw_frame_num
        if self.draw_frame_num:
            self.frame_num_annotator = FrameNumberAnnotator() 

        if save_tracks_dir:
            self.save_tracks_dir = save_tracks_dir
            self.writer = TracksJsonWriter(save_tracks_dir)
       
        field_image = cv2.imread(field_img_path)
        # Convert the field image to grayscale (black and white)
        field_image = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)

        # Convert grayscale back to 3 channels (since the main frame is 3-channel)
        field_image = cv2.cvtColor(field_image, cv2.COLOR_GRAY2BGR)

        # Initialize the speed estimator with the field image's dimensions
        self.speed_estimator = SpeedEstimator(field_image.shape[1], field_image.shape[0])
        
        self.frame_num = 0

        self.field_image = field_image

    def process(self, frames: List[np.ndarray], fps: float = 1e-6) -> None:
        """
        Processes a batch of video frames, detects and tracks objects, assigns ball possession, and annotates the frames.

        Args:
            frames (List[np.ndarray]): List of video frames.
            fps (float): Frames per second of the video, used for speed estimation.

        Returns:
            List[np.ndarray]: A list of annotated video frames.
        """
        
        self.cur_fps = max(fps, 1e-6)

        # Detect objects and keypoints in all frames
        batch_obj_detections = self.obj_tracker.detect(frames)
        batch_kp_detections = self.kp_tracker.detect(frames)

        # processed_frames = []

        # Process each frame in the batch
        for idx, (frame, object_detection, kp_detection) in enumerate(zip(frames, batch_obj_detections, batch_kp_detections)):
            
            # Track detected objects and keypoints
            obj_tracks = self.obj_tracker.track(object_detection)
            kp_tracks = self.kp_tracker.track(kp_detection)

            # Assign clubs to players based on their tracked position
            obj_tracks = self.club_assigner.assign_clubs(frame, obj_tracks)

            all_tracks = {'object': obj_tracks, 'keypoints': kp_tracks}

            # Map objects to a top-down view of the field
            # all_tracks = self.obj_mapper.map(all_tracks)

            # Estimate the speed of the tracked objects
            # all_tracks['object'] = self.speed_estimator.calculate_speed(
            #     all_tracks['object'], self.frame_num, self.cur_fps
            # )

            # Save tracking information if saving is enabled
            if self.save_tracks_dir:
                self._save_tracks(all_tracks)

            # self.frame_num += 1

            # Annotate the current frame with the tracking information
            # annotated_frame = self.annotate(frame, all_tracks)

            # Append the annotated frame to the processed frames list
            # processed_frames.append(annotated_frame)

        # return processed_frames
    
    def process_from_mot_export_and_kobj_json(self, frames: List[np.ndarray], fps: float = 1e-6, mot_file_path: str = "mot_results.txt", kobj_file_path: str = "") -> None:
        """
        Processes a batch of video frames, detects and tracks objects, assigns ball possession, and annotates the frames with track results from mot exports.

        Args:
            frames (List[np.ndarray]): List of video frames.
            fps (float): Frames per second of the video, used for speed estimation.
            file_path (str) : File path for mot exports.

        Returns:
            List[np.ndarray]: A list of annotated video frames.
        """
        
        self.cur_fps = max(fps, 1e-6)

        all_obj_tracks = {} # Clear existing data
        classes = ['ball', 'goalkeeper', 'player', 'referee']

        # Extract object tracks from mot
        with open(mot_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 9:
                    continue  # Skip malformed lines

                frame_num = int(parts[0])
                track_id = int(parts[1])
                x1 = float(parts[2])
                y1 = float(parts[3])
                width = float(parts[4])
                height = float(parts[5])
                conf = float(parts[6])
                class_id = int(parts[7])

                if class_id == 0:
                    continue

                class_name = classes[class_id]
                x2 = x1 + width
                y2 = y1 + height
                frame_idx = frame_num - 1  # Convert from 1-indexed to 0-indexed

                # Initialize nested dictionaries as needed
                if frame_idx not in all_obj_tracks:
                    all_obj_tracks[frame_idx] = {
                        "ball": {},
                        "goalkeeper": {},
                        "player": {},
                        "referee": {},
                    }

                all_obj_tracks[frame_idx][class_name][track_id] = {
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf
                }
        all_obj_tracks = [{**v} for _, v in all_obj_tracks.items()] 

        with open(kobj_file_path, 'r') as f:
            all_kp_tracks = json.load(f)

        processed_frames = []

        # Process each frame in the batch
        for idx, (frame, obj_tracks, kp_tracks) in enumerate(zip(frames, all_obj_tracks, all_kp_tracks)):
            # Assign clubs to players based on their tracked position
            obj_tracks = self.club_assigner.assign_clubs(frame, obj_tracks)
            kp_tracks = {
                int(k): (np.float32(v[0]), np.float32(v[1])) for k, v in kp_tracks.items()
            }       

            all_tracks = {'object': obj_tracks, 'keypoints': kp_tracks}

            # Map objects to a top-down view of the field
            all_tracks = self.obj_mapper.map(all_tracks)

            # Estimate the speed of the tracked objects
            all_tracks['object'] = self.speed_estimator.calculate_speed(
                all_tracks['object'], self.frame_num, self.cur_fps
            )

            # Save tracking information if saving is enabled
            if self.save_tracks_dir:
                self._save_tracks(all_tracks)

            self.frame_num += 1

            # Annotate the current frame with the tracking information
            annotated_frame = self.annotate(frame, all_tracks)

            # Append the annotated frame to the processed frames list
            processed_frames.append((self.frame_num, annotated_frame))

        return processed_frames

    
    def annotate(self, frame: np.ndarray, tracks: Dict) -> np.ndarray:
        """
        Annotates the given frame with analised data

        Args:
            frame (np.ndarray): The current video frame to be annotated.
            tracks (Dict[str, Dict[int, np.ndarray]]): A dictionary containing tracking data for objects and keypoints.

        Returns:
            np.ndarray: The annotated video frame.
        """
         
        # Draw the frame number if required
        if self.draw_frame_num:
            frame = self.frame_num_annotator.annotate(frame, {'frame_num': self.frame_num})
        
        # Annotate the frame with keypoint and object tracking information
        frame = self.kp_annotator.annotate(frame, tracks['keypoints'])
        frame = self.obj_annotator.annotate(frame, tracks['object'])
        
        # Project the object positions onto the football field image
        projection_frame = self.projection_annotator.annotate(self.field_image, tracks['object'], self.frame_num, self.save_tracks_dir)

        # Combine the frame and projection into a single canvas
        combined_frame = self._combine_frame_projection(frame, projection_frame)

        return combined_frame
    
    def export_mot(self, name: str = ""):
        os.makedirs('./output_videos/mot_results', exist_ok=True)
        self.obj_tracker.export_to_mot(save_path = f"./output_videos/mot_results/{name}.txt")

    def extract_speed(self):
        all_speed_history, all_distance_history = self.speed_estimator.extract_players_data()

        speed_history_path = os.path.join(self.save_tracks_dir, 'speed.txt')
        with open(speed_history_path, 'w') as f:
            for key, value in sorted(all_speed_history.items()):
                f.write(f'Track {key}: {value}\n')

        distance_history_path = os.path.join(self.save_tracks_dir, 'distance.txt')
        with open(distance_history_path, 'w') as f:
            for key, value in sorted(all_distance_history.items()):
                f.write(f'Track {key}: {value}\n')

    def import_mot(self, file_path: str = ""):
        self.obj_tracker.import_from_mot(file_path = file_path)

    def _combine_frame_projection(self, frame: np.ndarray, projection_frame: np.ndarray) -> np.ndarray:
        """
        Combines the original video frame with the projection of player positions on the field image.

        Args:
            frame (np.ndarray): The original video frame.
            projection_frame (np.ndarray): The projected field image with annotations.

        Returns:
            np.ndarray: The combined frame.
        """
        # Target canvas size
        canvas_width, canvas_height = 1920, 1080
        
        # Get dimensions of the original frame and projection frame
        h_frame, w_frame, _ = frame.shape
        h_proj, w_proj, _ = projection_frame.shape

        # Scale the projection to 70% of its original size
        scale_proj = 0.7
        new_w_proj = int(w_proj * scale_proj)
        new_h_proj = int(h_proj * scale_proj)
        projection_resized = cv2.resize(projection_frame, (new_w_proj, new_h_proj))

        # Create a blank canvas of 1920x1080
        combined_frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Copy the main frame onto the canvas (top-left corner)
        combined_frame[:h_frame, :w_frame] = frame

        # Set the position for the projection frame at the bottom-middle
        x_offset = (canvas_width - new_w_proj) // 2
        y_offset = canvas_height - new_h_proj - 25  # 25px margin from bottom

        # Blend the projection with 75% visibility (alpha transparency)
        alpha = 0.75
        overlay = combined_frame[y_offset:y_offset + new_h_proj, x_offset:x_offset + new_w_proj]
        cv2.addWeighted(projection_resized, alpha, overlay, 1 - alpha, 0, overlay)

        return combined_frame
    
    def _display_possession_text(self, frame: np.ndarray, club1_width: int, club2_width: int,
                                  neutral_width: int, bar_x: int, bar_y: int, 
                                 possession_club1_text: str, possession_club2_text: str, 
                                 club1_color: Tuple[int, int, int], club2_color: Tuple[int, int, int]) -> None:
        """
        Helper function to display possession percentages for each club below the progress bar.

        Args:
            frame (np.ndarray): The frame where the text will be displayed.
            club1_width (int): Width of club 1's possession bar.
            club2_width (int): Width of club 2's possession bar.
            neutral_width (int): Width of the neutral possession area.
            bar_x (int): X-coordinate of the progress bar.
            bar_y (int): Y-coordinate of the progress bar.
            possession_club1_text (str): Text for club 1's possession percentage.
            possession_club2_text (str): Text for club 2's possession percentage.
            club1_color (tuple): BGR color of club 1.
            club2_color (tuple): BGR color of club 2.
        """
        # Text for club 1
        club1_text_x = bar_x + club1_width // 2 - 10  # Center of club 1's possession bar
        club1_text_y = bar_y + 35  # 20 pixels below the bar
        cv2.putText(frame, possession_club1_text, (club1_text_x, club1_text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
        cv2.putText(frame, possession_club1_text, (club1_text_x, club1_text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, club1_color, 1)  # Club 1's color

        # Text for club 2
        club2_text_x = bar_x + club1_width + neutral_width + club2_width // 2 - 10  # Center of club 2's possession bar
        club2_text_y = bar_y + 35  # 20 pixels below the bar
        cv2.putText(frame, possession_club2_text, (club2_text_x, club2_text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
        cv2.putText(frame, possession_club2_text, (club2_text_x, club2_text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, club2_color, 1)  # Club 2's color



    def _save_tracks(self, all_tracks: Dict[str, Dict[int, np.ndarray]]) -> None:
        """
        Saves the tracking information for objects and keypoints to the specified directory.

        Args:
            all_tracks (Dict[str, Dict[int, np.ndarray]]): A dictionary containing tracking data for objects and keypoints.
        """
        self.writer.write(self.writer.get_object_tracks_path(), all_tracks['object'])
        self.writer.write(self.writer.get_keypoints_tracks_path(), all_tracks['keypoints'])
