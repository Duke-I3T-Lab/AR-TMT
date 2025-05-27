import numpy as np
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import warnings
import os
from scipy.spatial.transform import Rotation as R


# Image is width * height 380 * 355
OFFSETS = {
    "Catheter": np.array([80, 155]),
    "Ventricle": np.array([0, 0]),
    "Distance": np.array([0, 0]),
    "Angle": np.array([0, 0]),
    "HR": np.array([0, 0]),
}

REFERENCE_POINTS = {
    "Catheter": np.array([352, 355 - 2]) + OFFSETS["Catheter"],
    "Ventricle": np.array([184, 355 - 218]) + OFFSETS["Ventricle"],
    "Distance": np.array([190, 355 - 99]) + OFFSETS["Distance"],
    "Angle": np.array([307, 355 - 154]) + OFFSETS["Angle"],
    "HR": np.array([104, 355 - 218]) + OFFSETS["HR"],
}

RATIO_HOLOLENS_TO_IMAGE = 96 / 0.08
MARKERS = ["o", "s", "v", "^", "D", "P"]
COLORS = ["red", "blue", "green", "orange", "purple", "brown"]


def preprocess_target_to_int(
    df, use_one_hot=False, smooth_targets=True, trial_number=1
):
    target_to_roi = {
        "Catheter_Middle": 1,
        "lateral_ventricle": 2,
        "third_ventricle": 2,
        "R_foramen_of_monro": 2,
        "L_foramen_of_monro": 2,
        "cerebral_aqueduct": 2,
        "Target_Distance": 3 if trial_number not in [1, 6] else 0,
        "Tool_Angle": 4 if trial_number not in [1, 6] else 0,
        "HeartRate": 5 if trial_number in [3, 5, 6] else 0,
    }

    def convert_zeros(array, sublist_length=5):
        result = np.empty_like(array)
        for i, x in enumerate(array):
            if x != 0:
                result[i] = x
            else:
                neighbor = array[
                    max(0, i - sublist_length // 2) : min(
                        len(array), i + sublist_length // 2 + 1
                    )
                ]
                max_value = np.bincount(neighbor).argmax()
                if np.count_nonzero(neighbor == max_value) >= sublist_length // 2 + 1:
                    result[i] = max_value
                else:
                    result[i] = 0
        return result

    # map df['target'] using the dict into a column of integers
    df["target"] = df["target"].map(target_to_roi).fillna(0).astype("int64")
    # clean the data. 0's in between the same target should be filled with the same value
    if smooth_targets:
        df["target"] = convert_zeros(df["target"].to_numpy())
    # change to one hot
    if not use_one_hot:
        return df
    target_numpy = list(df["target"].to_numpy())
    target_one_hot = pd.get_dummies(target_numpy + list(range(6))).astype("float64")
    # drop the added ones
    target_one_hot = target_one_hot.iloc[: len(df)]
    target_one_hot.columns = [
        "other",
        "catheter",
        "ventricle",
        "distance",
        "angle",
        "heart_rate",
    ]
    # put it back to df
    df = pd.concat([df, target_one_hot], axis=1)
    df.drop("target", axis=1, inplace=True)

    return df


def clean_df_to_ROI_only(df):
    start_size = len(df)
    df = df[df["target"] != 0].copy()
    proportion = len(df) / start_size
    if proportion < 0.0:
        return None

    # get the target's position based on the 'target' column
    target_index = df["target"].astype(int).to_numpy() - 1
    # get a new column target_position by choosing from the five Position columns
    target_position = [
        df[f"{list(REFERENCE_POINTS.keys())[target_index[i]]}Position"].iloc[i]
        for i in range(len(target_index))
    ]
    df["target_position"] = target_position
    df.drop(
        [
            "CatheterPosition",
            "VentriclePosition",
            "DistancePosition",
            "AnglePosition",
            "HRPosition",
        ],
        axis=1,
        inplace=True,
    )
    split_columns_and_save(df, "target_position", split_num=3)
    # compute relative position
    df["relative_position_x"] = df["hit_position_x"] - df["target_position_x"]
    df["relative_position_y"] = df["hit_position_y"] - df["target_position_y"]
    df["relative_position_z"] = df["hit_position_z"] - df["target_position_z"]
    df.drop(
        ["target_position_x", "target_position_y", "target_position_z"],
        axis=1,
        inplace=True,
    )
    df.drop(
        ["hit_position_x", "hit_position_y", "hit_position_z"], axis=1, inplace=True
    )

    return df
    # df["target_position"] = df[f"{list(REFERENCE_POINTS.keys())[target_index]}Position"]


def split_columns_and_save(feature_df, col, split_num=3):
    if split_num == 3:
        feature_df[[f"{col}_x", f"{col}_y", f"{col}_z"]] = (
            feature_df[col].str.strip("()").str.split("/ ", expand=True)
        )
    elif split_num == 4:
        feature_df[[f"{col}_x", f"{col}_y", f"{col}_z", f"{col}_o"]] = (
            feature_df[col].str.strip("()").str.split("/ ", expand=True)
        )
    feature_df[f"{col}_x"] = pd.to_numeric(feature_df[f"{col}_x"])
    feature_df[f"{col}_y"] = pd.to_numeric(feature_df[f"{col}_y"])
    feature_df[f"{col}_z"] = pd.to_numeric(feature_df[f"{col}_z"])
    if split_num == 4:
        feature_df[f"{col}_o"] = pd.to_numeric(feature_df[f"{col}_o"])
    feature_df.drop(col, axis=1, inplace=True)
    return feature_df


def read_and_plot_2D_gaze(file_path, clean=True):
    trial_number = int(file_path[-5])
    data = pd.read_csv(
        file_path,
        skiprows=1,
        usecols=[
            "HitInfoTarget",
            "HitPosition",
            "Unnamed: 7",
            "Unnamed: 8",
            "Label",
            "ToolPosition",
            "HeadPosition",
            "DistPosition",
            "AnglePosition",
            "HRPosition",
            "HROrientation",
        ],
        dtype={
            "HitInfoTarget": str,
            "HitPosition": float,
            "Unnamed: 7": float,
            "Unnamed: 8": float,
            "Label": int,
            "ToolPosition": str,
            "HeadPosition": str,
            "DistPosition": str,
            "AnglePosition": str,
            "HRPosition": str,
            "HROrientation": str,
        },
    )
    data = data.rename(
        columns={
            "HitInfoTarget": "target",
            "HitPosition": "hit_position_x",
            "Unnamed: 7": "hit_position_y",
            "Unnamed: 8": "hit_position_z",
            "Label": "phase",
            "ToolPosition": "CatheterPosition",
            "HeadPosition": "VentriclePosition",
            "DistPosition": "DistancePosition",
            "HROrientation": "orientation",
        }
    )

    data = preprocess_target_to_int(data, trial_number=trial_number)
    data = clean_df_to_ROI_only(data)
    data = split_columns_and_save(data, "orientation", split_num=4)
    # draw for each phase

    for label in range(4, 6):
        # successively deal with each target, find indexes of 'target' being changed
        df = data[data["phase"] == label].copy()
        target_diff = df["target"].diff()
        changed_index = np.where(target_diff != 0)[0]
        if len(changed_index) <= 2:
            continue

        # deal with unchanged periods one by one
        start_index = 0
        all_nodes = None
        for ind in changed_index[1:]:
            df_slice = df.iloc[start_index:ind]
            start_index = ind
            target, _2d_coords = compute_2D_coordinates(
                df_slice
            )  # in terms of hololens coordinates
            # need to convert in onto the figure
            _2d_coords = (
                _2d_coords * RATIO_HOLOLENS_TO_IMAGE
                + list(REFERENCE_POINTS.values())[target - 1]
            )

            all_nodes = (
                _2d_coords if all_nodes is None else np.vstack((all_nodes, _2d_coords))
            )
            # plt.scatter(
            #     _2d_coords[:, 0],
            #     _2d_coords[:, 1],
            #     marker="o",
            #     facecolors="none",
            #     edgecolors=COLORS[target - 1],
            #     # label=list(REFERENCE_POINTS.keys())[target - 1],
            # )

            # scatter plot

        plt.plot(
            all_nodes[:, 0],
            all_nodes[:, 1],
            "-o",
            linestyle="-",
            # markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1,
        )

        # add an image as the background
        img = plt.imread("offline\setup_image.png")
        plt.imshow(img, zorder=0, extent=[0, 380, 0, 355])
        plt.xlim(-100, 480)
        plt.ylim(-100, 455)
        plt.xlabel("x coordinate")
        plt.ylabel("y coordinate")
        plt.title(f"Gaze Path at phase {'Holding' if label == 4 else 'Insertion'}")
        plt.legend()
        plt.grid(False)
        plt.axis("off")
        plt.show()


def compute_2D_coordinates(df):
    target = df["target"].to_numpy()[0]
    _3d_gaze_numpy = df[
        ["relative_position_x", "relative_position_y", "relative_position_z"]
    ].to_numpy()
    quaternion = df[
        ["orientation_x", "orientation_y", "orientation_z", "orientation_o"]
    ].to_numpy()
    _2d_gaze_numpy = []
    for i in range(len(quaternion)):
        r = R.from_quat(quaternion[i])
        m = r.as_matrix()
        _2d_gaze_numpy.append((m.T @ _3d_gaze_numpy[i])[:2])
    return target, np.array(_2d_gaze_numpy)


if __name__ == "__main__":
    read_and_plot_2D_gaze("dataset\Participant Results Cleaned/18/3.csv")
