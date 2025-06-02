import os
import numpy as np
import pandas as pd
import json
import re
from typing import List
from offline.data.gaze_data import GazeData
from sklearn.preprocessing import StandardScaler


class ARTMTGazeData(GazeData):
    def __init__(self, file_path: str) -> None:
        super().__init__(file_path=file_path)
        self.load_data(file_path)

    def load_data(self, file_path: str) -> None:
        """Load ARTMT eyetracking data from JSON file"""

        # Extract user ID and stage from file path
        folder_name = os.path.basename(os.path.dirname(file_path))
        match = re.search(r"task(\d+)", os.path.basename(file_path))
        stage_num = int(match.group(1)) if match else None

        # Read and process JSON data
        with open(file_path, "r", encoding="utf-8") as file:
            raw_data = file.read()
        fixed_json = "[" + raw_data.replace("}\n{", "},\n{") + "]"
        data = json.loads(fixed_json)
        df = pd.DataFrame(data)

        # Filter data where video has started
        if "Startvideo" in df.columns:
            df = df[df["Startvideo"] == 2].copy()
            df.reset_index(drop=True, inplace=True)

        if len(df) == 0:
            raise ValueError(f"No valid data found in {file_path}")

        # Extract timestamp and normalize to start at 0
        initial_time = (
            df["GazeData_read"].iloc[0]["Timestamp"]
            if "Timestamp" in df["GazeData_read"].iloc[0]
            else 0
        )
        initial_time = 0
        df.loc[:, "TimeStamp"] = df["GazeData_read"].apply(
            lambda x: x["Timestamp"] - initial_time if "Timestamp" in x else None
        )

        df = self.clean_data_by_timestamp(df)

        # Extract eye gaze position
        df["EyeGazePosition_x"] = df["GazeData_from"].apply(
            lambda x: x["EyeGazePosition"]["x"] if "EyeGazePosition" in x else None
        )
        df["EyeGazePosition_y"] = df["GazeData_from"].apply(
            lambda x: x["EyeGazePosition"]["y"] if "EyeGazePosition" in x else None
        )
        df["EyeGazePosition_z"] = df["GazeData_from"].apply(
            lambda x: x["EyeGazePosition"]["z"] if "EyeGazePosition" in x else None
        )

        # Extract fixation point
        df["EyeDirection_x"] = df["GazeData_read"].apply(
            lambda x: x["GazeDirection"]["x"] if "GazeDirection" in x else None
        )
        df["EyeDirection_y"] = df["GazeData_read"].apply(
            lambda x: x["GazeDirection"]["y"] if "GazeDirection" in x else None
        )
        df["EyeDirection_z"] = df["GazeData_read"].apply(
            lambda x: x["GazeDirection"]["z"] if "GazeDirection" in x else None
        )

        df["LeftEyeOpenness"] = df["GeometricData"].apply(
            lambda x: (
                next(
                    (
                        entry.get("EyeOpenness")
                        for entry in x
                        if entry.get("Eye") == "Left" and entry.get("Valid", False)
                    ),
                    None,
                )
                if isinstance(x, list)
                else None
            )
        )

        df["RightEyeOpenness"] = df["GeometricData"].apply(
            lambda x: (
                next(
                    (
                        entry.get("EyeOpenness")
                        for entry in x
                        if entry.get("Eye") == "Right" and entry.get("Valid", False)
                    ),
                    None,
                )
                if isinstance(x, list)
                else None
            )
        )

        df["IsBlink"] = df["GazeBehaviorData"].apply(
            lambda x: (
                x["GazeBehaviorType"].startswith("Blink")
                if "GazeBehaviorType" in x
                else False
            )
        )
        # enforce the last 2 indices to not be blinks
        df["IsBlink"].iloc[-2:] = False

        self.blink_indices = np.array([])
        if "IsBlink" in df.columns:
            self.blink_indices = df.index[df["IsBlink"]].to_numpy()
            # print the blink indices for debugging
            print(f"Blink indices found: {len(self.blink_indices)}")

        # Extract pupil diameter data
        df["LeftPupilDiameter"] = df["PupilData"].apply(
            lambda x: (
                next(
                    (
                        entry.get("PupilDiameter")
                        for entry in x
                        if entry.get("Eye") == "Left" and entry.get("Valid", False)
                    ),
                    None,
                )
                if isinstance(x, list)
                else None
            )
        )
        df["RightPupilDiameter"] = df["PupilData"].apply(
            lambda x: (
                next(
                    (
                        entry.get("PupilDiameter")
                        for entry in x
                        if entry.get("Eye") == "Right" and entry.get("Valid", False)
                    ),
                    None,
                )
                if isinstance(x, list)
                else None
            )
        )

        df = self.process_pupil_diameter(df)

        # Set up base class attributes
        self.start_timestamp = df["TimeStamp"].values

        # print the average interval between two timestamps
        self.interval = np.mean(np.diff(self.start_timestamp))
        print(f"Average interval between two timestamps: {self.interval:.4f} seconds")

        # Extract gaze direction (use normalized positions)
        gaze_directions = []
        for idx, row in df.iterrows():
            # Use fixation point as gaze direction
            direction = np.array(
                [
                    row["EyeDirection_x"] if row["EyeDirection_x"] is not None else 0,
                    row["EyeDirection_y"] if row["EyeDirection_y"] is not None else 0,
                    row["EyeDirection_z"] if row["EyeDirection_z"] is not None else 0,
                ]
            )
            # Normalize the direction vector
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            gaze_directions.append(direction)

        self.gaze_direction = np.array(gaze_directions)

        self.eye_position = df[
            ["EyeGazePosition_x", "EyeGazePosition_y", "EyeGazePosition_z"]
        ].to_numpy()

        # Extract pupil diameters
        self.left_pupil_diameter = df["LeftPupilDiameter"].fillna(0).values
        self.right_pupil_diameter = df["RightPupilDiameter"].fillna(0).values

        self.label = [1] * len(df)
        self.label = np.array(self.label)

        # Set up indices
        self.indices = np.arange(len(df))
        self.first_index = 0

        # Store raw data for debugging
        self.raw_df = df

    def process_pupil_diameter(self, df):
        # first, all pupil diameter should be between 0.004 and 0.009, clamp them if not
        df["LeftPupilDiameter"] = df["LeftPupilDiameter"].clip(0.0015, 0.009)
        df["RightPupilDiameter"] = df["RightPupilDiameter"].clip(0.0015, 0.009)
        # then extract to numpy array
        self.left_pupil_diameter = df["LeftPupilDiameter"].to_numpy()
        self.right_pupil_diameter = df["RightPupilDiameter"].to_numpy()
        return df

    def clean_data_by_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        # Create a copy to work with
        cleaned_df = df.copy()

        # Get timestamp values
        timestamps = cleaned_df["TimeStamp"].values

        # Find indices where timestamp decreases (not in ascending order)
        indices_to_remove = []
        i = 1
        while i < len(timestamps):
            if timestamps[i] < timestamps[i - 1]:
                # Found a decrease, mark this index for removal
                start_removal = i

                # Continue removing until we find a timestamp greater than the last valid one
                last_valid_timestamp = timestamps[i - 1]
                while i < len(timestamps) and timestamps[i] <= last_valid_timestamp:
                    indices_to_remove.append(i)
                    i += 1
            else:
                i += 1

        # Remove the problematic indices
        if indices_to_remove:
            cleaned_df = cleaned_df.drop(
                cleaned_df.index[indices_to_remove]
            ).reset_index(drop=True)
            print(
                f"Removed {len(indices_to_remove)} rows due to non-ascending timestamps"
            )

        return cleaned_df

    def normalize_pupil_diameter(
        self,
        normalizer_left: StandardScaler,
        normalizer_right: StandardScaler,
        use_normal_dist=False,
    ) -> None:
        if use_normal_dist:
            self.left_pupil_diameter = normalizer_left.transform(
                self.left_pupil_diameter.reshape(-1, 1)
            ).flatten()

            self.right_pupil_diameter = normalizer_right.transform(
                self.right_pupil_diameter.reshape(-1, 1)
            ).flatten()

            # print min and max of the normalized right pupil diameter
        else:
            self.left_pupil_diameter = (
                self.left_pupil_diameter - normalizer_left.mean_
            ) / normalizer_left.mean_
            self.df["LeftPupilDiameter"] = (
                self.df["LeftPupilDiameter"] - normalizer_left.mean_
            ) / normalizer_left.mean_
            self.right_pupil_diameter = (
                self.right_pupil_diameter - normalizer_right.mean_
            ) / normalizer_right.mean_
            self.df["RightPupilDiameter"] = (
                self.df["RightPupilDiameter"] - normalizer_right.mean_
            ) / normalizer_right.mean_

    def get_total_duration(self) -> float:
        """Get the total duration of the data in seconds"""
        return self.start_timestamp[-1] - self.start_timestamp[0]
