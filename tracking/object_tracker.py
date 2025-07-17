from tracking.abstract_tracker import AbstractTracker

import supervision as sv
import cv2
from typing import List
import numpy as np
from ultralytics.engine.results import Results, Boxes
from ultralytics import YOLO
import torch
import copy

# No longer inheriting AbstractTracker
class ObjectTracker():

    def __init__(self, player_model_path: str, ball_model_path: str, conf: float = 0.5, ball_conf: float = 0.3) -> None:
        """
        Initialize ObjectTracker with detection and tracking.

        Args:
            model_path (str): Model Path.
            conf (float): Confidence threshold for detection.
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.player_model = YOLO(player_model_path)
        self.player_model.to(device)
        self.ball_model = YOLO(ball_model_path)
        self.ball_model.to(device)

        self.conf = conf
        self.ball_conf = ball_conf
        self.classes = ['ball', 'goalkeeper', 'player', 'referee']
        self.tracker = sv.ByteTrack(
            lost_track_buffer=30,
            frame_rate=25,
        )  # Initialize ByteTracker
        self.tracker.reset()
        # self.all_tracks = {class_name: {} for class_name in self.classes}  # Initialize tracks
        self.all_tracks = {}
        self.cur_frame = 0  # Frame counter initialization
        self.original_size = (1920, 1080)  # Original frame size (1920x1080)
        self.scale_x = self.original_size[0] / 1280
        self.scale_y = self.original_size[1] / 1280

    def detect(self, frames: List[np.ndarray]) -> List[Results]:
        """
        Perform object detection on multiple frames.

        Args:
            frames (List[np.ndarray]): List of frames to perform object detection on.

        Returns:
            List[Results]: Detection results for each frame.
        """
        # Preprocess: Resize frames to 1280x1280
        resized_frames = [self._preprocess_frame(frame) for frame in frames]

        # Use YOLOv8's predict method to handle batch inference
        player_detections = self.player_model.predict(resized_frames, conf=self.conf, verbose=False)
        ball_detections = self.ball_model.predict(resized_frames, conf=self.ball_conf, verbose=False)

        # TODO: Gabungin model 3+1 nya, append detections.boxesnya, classnya disesuaikan sesuai dengan model 4, tambahin waktu di speednya, adjust names nya
        detections = copy.deepcopy(player_detections)
        names = {
            0: 'ball',
            1: 'goalkeeper',
            2: 'player',
            3: 'referee',
        }

        for i in range(len(detections)):
            detections[i].names = names # Adjust detections results to 4 class

            player_boxes = player_detections[i].boxes
            ball_boxes = ball_detections[i].boxes
            player_data = player_boxes.data.clone()
            ball_data = ball_boxes.data.clone()

            player_xyxy = player_data[:, 0:4]
            player_conf = player_data[:, 4]
            player_cls = player_data[:, 5]
            class_mapping = torch.tensor([1, 2, 3], device=player_cls.device)
            player_cls = class_mapping[player_cls.long()] # Remapped player class to adjust to original classes
            player_new_boxes = torch.cat([
                player_xyxy, 
                player_conf.unsqueeze(1), 
                player_cls.unsqueeze(1)
            ], dim=1)

            ball_xyxy = ball_data[:, 0:4]
            ball_conf = ball_data[:, 4]
            ball_cls = ball_data[:, 5] # Keep class becausae it's already 0 (same as original classes)
            ball_new_boxes = torch.cat([
                ball_xyxy, 
                ball_conf.unsqueeze(1), 
                ball_cls.unsqueeze(1)
            ], dim=1)

            # Only take 1 ball detection that is the highest
            # if ball_new_boxes.shape[0] > 0:
            #     sorted_ball_boxes = ball_new_boxes[ball_new_boxes[:, 4].argsort(descending=True)]
            #     top_ball_box = sorted_ball_boxes[0].unsqueeze(0)  # shape [1, 6]
            # else:
            #     top_ball_box = torch.empty((0, 6), device=ball_new_boxes.device)
            top_ball_box = torch.empty((0, 6), device=ball_new_boxes.device)


            new_data = torch.cat([player_new_boxes, top_ball_box], dim=0)
            sorted_indices = new_data[:, 4].argsort(descending=True) # sort detections based on it's confidence score
            new_data = new_data[sorted_indices]
            orig_shape = detections[i].orig_shape
            new_boxes = Boxes(new_data, orig_shape=orig_shape)

            detections[i].boxes = new_boxes

            # Add inference time for both models
            combined_speed = {}
            for key in detections[i].speed:
                combined_speed[key] = (
                    player_detections[i].speed[key] +
                    ball_detections[i].speed[key]
                )
            detections[i].speed = combined_speed

        return detections  # Batch of detections

    def track(self, detection: Results) -> dict:
        """
        Perform object tracking on detection.

        Args:
            detection (Results): Detected objects for a single frame.

        Returns:
            dict: Dictionary containing tracks of the frame.
        """
        # Convert Ultralytics detections to supervision
        detection_sv = sv.Detections.from_ultralytics(detection)

        # Perform ByteTracker object tracking on the detections
        tracks = self.tracker.update_with_detections(detection_sv)

        self.current_frame_tracks = self._tracks_mapper(tracks, self.classes)
        
        # Store the current frame's tracking information in all_tracks
        self.all_tracks[self.cur_frame] = self.current_frame_tracks.copy()

        # Increment the current frame counter
        self.cur_frame += 1

        # Return only the last frame's data
        return self.current_frame_tracks
    
    def export_to_mot(self, save_path: str = "mot_results.txt"):
        """
        Export tracking results to MOT Challenge format.
        
        Args:
            save_path (str): Path to save the MOT format results file.
        """
        with open(save_path, 'w') as f:
            for frame_idx, frame_data in self.all_tracks.items():
                frame_num = frame_idx + 1  # MOT format is 1-indexed

                for class_name, tracks in frame_data.items():
                    class_id = self.classes.index(class_name)

                    for track_id, info in tracks.items():
                        x1, y1, x2, y2 = info['bbox']
                        conf = info['conf']
                        width = x2 - x1
                        height = y2 - y1

                        line = f"{frame_num},{track_id},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},{conf:.2f},{class_id},-1,-1\n"
                        f.write(line)

    # def import_from_mot(self, file_path: str = "mot_results.txt"):
    #     """
    #     Import tracking results from a MOT Challenge format file and reconstruct self.all_tracks.
        
    #     Args:
    #         file_path (str): Path to the MOT format results file.
    #     """
    #     self.all_tracks = {}  # Clear existing data

    #     with open(file_path, 'r') as f:
    #         for line in f:
    #             parts = line.strip().split(',')
    #             if len(parts) < 9:
    #                 continue  # Skip malformed lines

    #             frame_num = int(parts[0])
    #             track_id = int(parts[1])
    #             x1 = float(parts[2])
    #             y1 = float(parts[3])
    #             width = float(parts[4])
    #             height = float(parts[5])
    #             conf = float(parts[6])
    #             class_id = int(parts[7])

    #             class_name = self.classes[class_id]
    #             x2 = x1 + width
    #             y2 = y1 + height
    #             frame_idx = frame_num - 1  # Convert from 1-indexed to 0-indexed

    #             # Initialize nested dictionaries as needed
    #             if frame_idx not in self.all_tracks:
    #                 self.all_tracks[frame_idx] = {}
    #             if class_name not in self.all_tracks[frame_idx]:
    #                 self.all_tracks[frame_idx][class_name] = {}

    #             self.all_tracks[frame_idx][class_name][track_id] = {
    #                 "bbox": [x1, y1, x2, y2],
    #                 "conf": conf
    #             }

    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the frame by resizing it to 1280x1280.

        Args:
            frame (np.ndarray): The input image frame.

        Returns:
            np.ndarray: The resized frame.
        """
        # Resize the frame to 1280x1280
        resized_frame = cv2.resize(frame, (1280, 1280))
        return resized_frame
    
    def _tracks_mapper(self, tracks: sv.Detections, class_names: List[str]) -> dict:
        """
        Maps tracks to a dictionary by class and tracker ID. Also, adjusts bounding boxes to 1920x1080 resolution.

        Args:
            tracks (sv.Detections): Tracks from the frame.
            class_names (List[str]): List of class names.

        Returns:
            dict: Mapped detections for the frame.
        """
        # Initialize the dictionary
        result = {class_name: {} for class_name in class_names}

        # Extract relevant data from tracks
        xyxy = tracks.xyxy  # Bounding boxes
        class_ids = tracks.class_id  # Class IDs
        tracker_ids = tracks.tracker_id  # Tracker IDs
        confs = tracks.confidence

        # Iterate over all tracks
        for bbox, class_id, track_id, conf in zip(xyxy, class_ids, tracker_ids, confs):
            class_name = class_names[class_id]

            # Skip balls with confidence lower than ball_conf
            if class_name == "ball" and conf < self.ball_conf:
                continue  # Skip low-confidence ball detections

            # Create class_name entry if not already present
            if class_name not in result:
                result[class_name] = {}

            # Scale the bounding box back to the original resolution (1920x1080)
            scaled_bbox = [
                bbox[0] * self.scale_x,  # x1
                bbox[1] * self.scale_y,  # y1
                bbox[2] * self.scale_x,  # x2
                bbox[3] * self.scale_y   # y2
            ]

            # Add track_id entry if not already present
            if track_id not in result[class_name]:
                result[class_name][track_id] = {'bbox': scaled_bbox, 'conf': conf}

        return result
