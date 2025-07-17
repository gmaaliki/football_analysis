import re
import numpy as np
import pandas as pd

class CalculatePlayerStats:
    def __init__(self, save_tracks_dir: str):
        self.save_tracks_dir = save_tracks_dir
        self.distance_file = f"{save_tracks_dir}/distance.txt"
        self.speed_file = f"{save_tracks_dir}/speed.txt"
    
    def distance_stat(self):
        with open(self.distance_file, "r") as file:
            raw_text = file.read()
        track_sum = []
        for match in re.finditer(r"Track (\d+): \[(.*?)\]", raw_text, re.DOTALL):
            track_id = int(match.group(1))
            values = list(map(float, match.group(2).split(",")))
            total_distance = np.sum(values) if values else 0.0
            track_sum.append((track_id, total_distance))
        df_avg_distance = pd.DataFrame(track_sum, columns=["Track ID", "Total Distance"])

        df_avg_distance.to_csv(f"{self.save_tracks_dir}/sum_distance_per_track.csv", index=False)

    def speed_stat(self):
        with open(self.speed_file, "r") as file:
            raw_text = file.read()
        track_averages = []
        for match in re.finditer(r"Track (\d+): \[(.*?)\]", raw_text, re.DOTALL):
            track_id = int(match.group(1))
            values = list(map(float, match.group(2).split(",")))
            avg_speed = np.mean(values) if values else 0.0
            track_averages.append((track_id, avg_speed))

        df_avg_speed = pd.DataFrame(track_averages, columns=["Track ID", "Average Speed"])

        df_avg_speed.to_csv(f"{self.save_tracks_dir}/average_speed_per_track.csv", index=False)
