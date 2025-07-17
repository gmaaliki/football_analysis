from .abstract_annotator import AbstractAnnotator
from utils import is_color_dark, rgb_bgr_converter

import cv2
import os
import numpy as np
from typing import Dict


class ProjectionAnnotator(AbstractAnnotator):
    """
    Class to annotate projections on a projection image and different markers for ball, players, referees, and goalkeepers.
    """

    def _draw_outline(self, frame: np.ndarray, pos: tuple, shape: str = 'circle', size: int = 10, is_dark: bool = True) -> None:
        """
        Draws a white or black outline around the object based on its color and shape.
        
        Parameters:
            frame (np.ndarray): The image on which to draw the outline.
            pos (tuple): The (x, y) position of the object.
            shape (str): The shape of the outline ('circle', 'square', 'dashed_circle', 'plus').
            size (int): The size of the outline.
            is_dark (bool): Flag indicating whether the color is dark (determines outline color).
        """
        outline_color = (255, 255, 255) if is_dark else (0, 0, 0)

        if shape == 'circle':
            cv2.circle(frame, (int(pos[0]), int(pos[1])), radius=size + 2, color=outline_color, thickness=2)
        elif shape == 'square':
            top_left = (int(pos[0]) - (size + 2), int(pos[1]) - (size + 2))
            bottom_right = (int(pos[0]) + (size + 2), int(pos[1]) + (size + 2))
            cv2.rectangle(frame, top_left, bottom_right, color=outline_color, thickness=2)
        elif shape == 'dashed_circle':
            dash_length, gap_length = 30, 30
            for i in range(0, 360, dash_length + gap_length):
                start_angle_rad, end_angle_rad = np.radians(i), np.radians(i + dash_length)
                start_x = int(pos[0]) + int((size + 2) * np.cos(start_angle_rad))
                start_y = int(pos[1]) + int((size + 2) * np.sin(start_angle_rad))
                end_x = int(pos[0]) + int((size + 2) * np.cos(end_angle_rad))
                end_y = int(pos[1]) + int((size + 2) * np.sin(end_angle_rad))
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color=(0, 0, 0), thickness=2)
        elif shape == 'plus':
            cv2.line(frame, (int(pos[0]) - size, int(pos[1])), (int(pos[0]) + size, int(pos[1])), color=outline_color, thickness=10)
            cv2.line(frame, (int(pos[0]), int(pos[1]) - size), (int(pos[0]), int(pos[1]) + size), color=outline_color, thickness=10)


    def annotate(self, frame: np.ndarray, tracks: Dict, frame_num: int, save_tracks_dir: str) -> np.ndarray:
        """
        Annotates an image with projected player, goalkeeper, referee, and ball positions.
        
        Parameters:
            frame (np.ndarray): The image on which to draw the annotations.
            tracks (Dict): A dictionary containing tracking information for 'player', 'goalkeeper', 'referee', and 'ball'.

        Returns:
            np.ndarray: The annotated frame.
        """
        frame = frame.copy()

        for class_name, track_data in tracks.items():
            if class_name != 'ball':  # Ball is drawn later
                for track_id, track_info in track_data.items():
                    if 'projection' not in track_info:
                        continue
                    proj_pos = track_info['projection']
                    color = track_info.get('club_color', (255, 255, 255))
                    color = rgb_bgr_converter(color)
                    is_dark_color = is_color_dark(color)

                    if class_name in ['player', 'goalkeeper']:
                        shape = 'square' if class_name == 'goalkeeper' else 'circle'
                        self._draw_outline(frame, proj_pos, shape=shape, is_dark=is_dark_color)

                        if track_info.get('has_ball', False):
                            cv2.circle(frame, (int(proj_pos[0]), int(proj_pos[1])), radius=15, color=(0, 255, 0), thickness=2)
                        if shape == 'circle':
                            cv2.circle(frame, (int(proj_pos[0]), int(proj_pos[1])), radius=10, color=color, thickness=-1)
                        else:
                            top_left = (int(proj_pos[0]) - 10, int(proj_pos[1]) - 10)
                            bottom_right = (int(proj_pos[0]) + 10, int(proj_pos[1]) + 10)
                            cv2.rectangle(frame, top_left, bottom_right, color=color, thickness=-1)

                    elif class_name == 'referee':
                        self._draw_outline(frame, proj_pos, shape='dashed_circle', is_dark=is_dark_color)
        projection_output_dir = os.path.join(save_tracks_dir, "field_projection")
        os.makedirs(projection_output_dir, exist_ok=True)
        cv2.imwrite(f"{projection_output_dir}/{frame_num}.jpg", frame)

        return frame
    