import json
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from scipy.stats import mode

from offline.data import GazeData
from offline.modules import Module
import offline.utils as utils
import os


def discard_short_periods(signal, all_extracted_periods, short_fixation_threshold=0.06):
    original_length = len(all_extracted_periods)
    # print("Original length", original_length)
    cleaned_periods = [
        period
        for period in all_extracted_periods
        if signal.start_timestamp[period[1]] - signal.start_timestamp[period[0]]
        >= short_fixation_threshold
    ]
    all_short_durations = [
        round(
            (signal.start_timestamp[period[1]] - signal.start_timestamp[period[0]])
            * 1000,
            1,
        )
        for period in all_extracted_periods
        if signal.start_timestamp[period[1]] - signal.start_timestamp[period[0]]
        < short_fixation_threshold
    ]
    # print("Discarded", original_length - len(cleaned_periods), "periods")
    # print("duration of them: ", all_short_durations)
    return cleaned_periods


def find_all_periods(signal, indices=[], data=None):
    start = None
    all_periods = []
    for i in range(len(signal) - 1):
        if start is None:
            if signal[i] and indices[i] + 1 == indices[i + 1]:
                start = i
        else:  # there is a start
            if not signal[i] or indices[i] + 1 != indices[i + 1]:
                all_periods.append((start, i))
                # if len(indices) > 0 and indices[i] - indices[start] > 400:
                #     print("start", start, "end", i, "indices", indices[start], indices[i])
                #     print(indices[start:i+1])
                start = None
    if start is not None and indices[len(signal) - 1] - 1 == indices[len(signal) - 2]:
        all_periods.append((start, len(signal) - 1))
    return all_periods


class DurationDistanceVelocity(Module):
    def __init__(self, window_size=1) -> None:
        self.window_size = window_size

    def update(self, data: GazeData) -> GazeData:
        if self.window_size > 1:
            data.duration = np.roll(
                data.start_timestamp, -self.window_size // 2, axis=0
            ) - np.roll(data.start_timestamp, self.window_size // 2, axis=0)
            data.duration = data.duration[
                self.window_size // 2 : -self.window_size // 2
            ]
            data.distance = utils.angular_distance(
                np.roll(data.gaze_direction, -self.window_size // 2, axis=0),
                np.roll(data.gaze_direction, self.window_size // 2, axis=0),
            )[self.window_size // 2 : -self.window_size // 2]

            # eye_position_offset = np.roll(
            #     data.eye_position, -21 // 2, axis=0
            # ) - np.roll(data.eye_position, 21 // 2, axis=0)
            # eye_position_offset = eye_position_offset[21 // 2 : -21 // 2, :]
            # data.delta = utils.gaze_offset_direction(
            #     np.roll(data.gaze_direction, 21 // 2, axis=0)[21 // 2 : -21 // 2],
            #     np.roll(data.gaze_direction, -21 // 2, axis=0)[21 // 2 : -21 // 2],
            #     reference_directions=eye_position_offset,
            # )
            # data.delta = np.array(
            #     [-1] * ((21 - self.window_size) // 2)
            #     + list(data.delta)
            #     + [-1] * ((21 - self.window_size) // 2)
            # )

            # check for continuous indices
            for i in range(1, len(data.indices) - 2):
                if data.indices[i + 1] != data.indices[i] + 1:
                    # print("Non continuous indices at index", i, "current index is: ", data.indices[i], "next is: ", data.indices[i+1])
                    # print("Consecutive timestamps", data.start_timestamp[i], data.start_timestamp[i+1])
                    data.duration[i - self.window_size // 2] = (
                        data.start_timestamp[i]
                        - data.start_timestamp[i - self.window_size // 2]
                    )
                    data.distance[i - self.window_size // 2] = (
                        utils.single_angular_distance(
                            data.gaze_direction[i],
                            data.gaze_direction[i - self.window_size // 2],
                        )
                    )

            assert len(data.duration) == len(data.distance)
        else:
            data.duration = (
                np.roll(data.start_timestamp, -1, axis=0) - data.start_timestamp
            )[:-1]
            data.distance = utils.angular_distance(
                data.gaze_direction[:-1], data.gaze_direction[1:]
            )
            for i in range(len(data.duration)):
                if data.indices[i + 1] != data.indices[i] + 1:
                    data.duration[i] = (
                        data.start_timestamp[i + 1] - data.start_timestamp[i]
                    )
                    data.distance[i] = utils.single_angular_distance(
                        data.gaze_direction[i + 1], data.gaze_direction[i]
                    )
            # print the zeros in duration
        data.velocity = data.distance / data.duration
        # print("velocity: ", data.velocity)
        test = np.where(data.duration <= 0)

        # find where velocity is nan
        nan_indices = np.where(np.isnan(data.velocity))[0]
        for i in nan_indices:
            print("Nan velocity at index", i)
            print(
                "Start timestamp",
                data.start_timestamp[i - 2],
                "End timestamp",
                data.start_timestamp[i + 2],
            )
            print(
                "Gaze direction", data.gaze_direction[i - 2], data.gaze_direction[i + 2]
            )
            print("Distance", data.distance[i])
            print("Duration", data.duration[i])
        assert np.all(data.duration > 0)

        assert np.all(data.velocity >= 0)
        data.start_timestamp = (
            data.start_timestamp[:-1]
            if self.window_size == 1
            else data.start_timestamp[self.window_size // 2 : -self.window_size // 2]
        )
        data.gaze_direction = (
            data.gaze_direction[:-1]
            if self.window_size == 1
            else data.gaze_direction[self.window_size // 2 : -self.window_size // 2]
        )

        if hasattr(data, "left_amount"):
            data.left_amount = (
                data.left_amount[:-1]
                if self.window_size == 1
                else data.left_amount[self.window_size // 2 : -self.window_size // 2]
            )
            data.right_amount = (
                data.right_amount[:-1]
                if self.window_size == 1
                else data.right_amount[self.window_size // 2 : -self.window_size // 2]
            )
        if hasattr(data, "label"):
            data.label = (
                data.label[:-1]
                if self.window_size == 1
                else data.label[self.window_size // 2 : -self.window_size // 2]
            )

        if hasattr(data, "blink"):
            data.blink = (
                data.blink[:-1]
                if self.window_size == 1
                else data.blink[self.window_size // 2 : -self.window_size // 2]
            )

        if len(data.indices) > 0:
            data.indices = (
                data.indices[:-1]
                if self.window_size == 1
                else data.indices[self.window_size // 2 : -self.window_size // 2]
            )
        if hasattr(data, "blink_indices") and len(data.blink_indices) > 0:
            while data.blink_indices[-1] + data.first_index >= data.indices[-1]:
                data.blink_indices = data.blink_indices[:-1]
            if data.blink_indices[0] + data.first_index < data.indices[0]:
                data.blink_indices = data.blink_indices[1:]
        return data


class MobilityDetection(Module):
    def __init__(self, window_size=5, threshold=0.0028) -> None:
        super().__init__()
        self.window_size = window_size
        self.threshold = threshold

    def update(self, data: GazeData) -> GazeData:
        # if len(data.eye_position) > len(data.velocity):
        #     # eye position is longer, remove from start and end to match length
        #     remove_count = (len(data.eye_position) - len(data.velocity)) / 2
        #     data.eye_position = (
        #         data.eye_position[remove_count : -remove_count - 1]
        #         if remove_count > 0
        #         else data.eye_position
        #     )
        eye_position_offset = np.roll(
            data.eye_position, -self.window_size // 2, axis=0
        ) - np.roll(data.eye_position, self.window_size // 2, axis=0)

        eye_position_offset = eye_position_offset[
            self.window_size // 2 : -self.window_size // 2, :
        ]
        mobility_mask = (
            np.linalg.norm(eye_position_offset, axis=1)
            > self.threshold * self.window_size
        )

        if len(mobility_mask) < len(data.indices):
            padding_length = (len(data.indices) - len(mobility_mask)) // 2
            mobility_mask = (
                np.ones(padding_length).astype(bool).tolist()
                + list(mobility_mask)
                + np.ones(padding_length).astype(bool).tolist()
            )
            assert len(mobility_mask) == len(data.indices)
            data.eye_position = data.eye_position[
                self.window_size // 2
                - padding_length : -self.window_size // 2
                + padding_length
            ]
            assert len(data.eye_position) == len(data.indices)
        data.mobility_mask = mobility_mask
        return data


class OffsetVelocityThresholding(Module):
    def __init__(
        self, window_size=1, distance_upper_bound=1, distance_lower_bound=0.5
    ) -> None:
        """
        Upper and lower bounds are assumed distance from eye to target in meters.
        """
        self.window_size = window_size
        self.distance_upper_bound = distance_upper_bound
        self.distance_lower_bound = distance_lower_bound

    def update(self, data: GazeData) -> GazeData:

        if self.window_size > 1:
            data.eye_position_offset = np.roll(
                data.eye_position, -self.window_size // 2, axis=0
            ) - np.roll(data.eye_position, self.window_size // 2, axis=0)
            data.eye_position_offset = data.eye_position_offset[
                self.window_size // 2 : -self.window_size // 2, :
            ]

            data.start_direction = np.roll(
                data.gaze_direction, self.window_size // 2, axis=0
            )
            for i in range(1, len(data.indices) - 2):
                if data.indices[i + 1] != data.indices[i] + 1:
                    data.eye_position_offset[i - self.window_size // 2] = (
                        data.eye_position[i]
                        - data.eye_position[i - self.window_size // 2]
                    )
            data.eye_position_offset_distance = np.linalg.norm(
                data.eye_position_offset, axis=1
            )

        else:
            data.eye_position_offset = data.eye_position[1:] - data.eye_position[:-1]
            data.eye_position_offset_distance = np.linalg.norm(
                data.eye_position_offset, axis=1
            )
            data.start_direction = data.gaze_direction[1:]

        cosalpha = utils.cos_distance(data.eye_position_offset, data.start_direction)
        alpha = np.arccos(cosalpha)
        constraint_1_min = (
            data.eye_position_offset_distance
            * np.sin(alpha)
            / self.distance_upper_bound
        )
        constraint_1_max = (
            data.eye_position_offset_distance
            * np.sin(alpha)
            / self.distance_lower_bound
        )
        constraint_2_min = np.sqrt(
            np.sin(alpha)
            / (
                (self.distance_upper_bound / data.eye_position_offset_distance) ** 2
                - 2
                * cosalpha
                * self.distance_upper_bound
                / data.eye_position_offset_distance
                + 1
            )
        )
        constraint_2_max = np.sqrt(
            np.sin(alpha)
            / (
                (self.distance_lower_bound / data.eye_position_offset_distance) ** 2
                - 2
                * cosalpha
                * self.distance_lower_bound
                / data.eye_position_offset_distance
                + 1
            )
        )
        # take the max of two mins, and min of two maxs
        data.velocity_threshold_max = np.rad2deg(
            np.arcsin(np.minimum(constraint_1_max, constraint_2_max))
        )
        data.velocity_threshold_min = np.rad2deg(
            np.arcsin(np.maximum(constraint_1_min, constraint_2_min))
        )
        # divide by duration
        data.velocity_threshold_max = data.velocity_threshold_max / data.duration
        data.velocity_threshold_min = data.velocity_threshold_min / data.duration

        # where offset is 0, set both thresholds to 0
        data.velocity_threshold_max[
            np.where(data.eye_position_offset_distance == 0)
        ] = 0
        data.velocity_threshold_min[
            np.where(data.eye_position_offset_distance == 0)
        ] = 0
        data.eye_position = data.eye_position[
            self.window_size // 2 : -self.window_size // 2
        ]

        return data


class MedianFilter(Module):
    def __init__(self, attr: str, window_size: int = 5) -> None:
        self.attr = attr
        self.window_size = window_size

    def update(self, data: GazeData) -> GazeData:
        signal = getattr(data, self.attr)
        # signal = medfilt(signal, kernel_size=self.window_size)
        previous_signal = signal.copy()
        for i in range(len(signal) - self.window_size // 2 - 1):
            if self.window_size == 5 and data.indices[i + 2] - data.indices[i - 2] == 4:
                signal[i] = median_filter(previous_signal[i - 2 : i + 3], size=(5, 1))[
                    2
                ]
            elif data.indices[i + 1] - data.indices[i - 1] == 2:
                signal[i] = median_filter(previous_signal[i - 1 : i + 2], size=(5, 1))[
                    1
                ]

        setattr(data, self.attr, signal)

        return data


class ModeFilter(Module):
    def __init__(self, attr: str, window_size: int = 3) -> None:
        self.attr = attr
        self.window_size = window_size

    def update(self, data: GazeData) -> GazeData:
        signal = getattr(data, self.attr)
        # if true false signal, convert to 0, 1 signal
        signal = [1 if s else 0 for s in signal]
        for i in range(len(signal) - self.window_size):
            signal[i + self.window_size // 2] = mode(
                signal[i : i + self.window_size], keepdims=True
            )[0][0]
        setattr(data, self.attr, signal)

        return data


class ROIFilter(Module):
    def __init__(self, attr: str, window_size: int = 5) -> None:
        self.attr = attr
        self.window_size = window_size

    def update(self, data: GazeData) -> GazeData:
        signal = getattr(data, self.attr)
        for i in range(len(signal) - self.window_size):
            windowed_signal = signal[i : i + self.window_size]
            val = max(set(windowed_signal), key=list(windowed_signal).count)

            if val != "other":
                signal[i + self.window_size // 2] = val
        setattr(data, self.attr, signal)

        return data


class MovingAverageFilter(Module):
    def __init__(self, attr: str, window_size: int = 5) -> None:
        self.attr = attr
        self.window_size = window_size

    def update(self, data: GazeData) -> GazeData:
        signal = getattr(data, self.attr)
        for i in range(len(signal) - self.window_size):
            signal[i + self.window_size // 2] = np.mean(
                signal[i : i + self.window_size]
            )
        setattr(data, self.attr, signal)

        return data


class SavgolFilter(Module):
    def __init__(self, attr: str, window_size: int = 3, order: int = 1) -> None:
        self.attr = attr
        self.window_size = window_size
        self.order = order

    def update(self, data: GazeData) -> GazeData:
        signal = getattr(data, self.attr)
        signal = savgol_filter(
            signal, window_length=self.window_size, polyorder=self.order
        )
        setattr(data, self.attr, signal)
        return data


class AggregateFixations(Module):
    def __init__(
        self,
        short_fixation_threshold=0.06,
        merge_interval_threshold=0.075,
        merge_direction_threshold=0.5,
        target_threshold=0.5,
    ) -> None:
        super().__init__()
        self.short_fixation_threshold = short_fixation_threshold
        self.merge_interval_threshold = merge_interval_threshold
        self.merge_direction_threshold = merge_direction_threshold
        self.total_fixation_duration = 0
        self.target_threshold = target_threshold

    def merge_fixation_periods(
        self,
        all_extracted_periods,
        merge_interval_threshold=0.075,
        merge_direction_threshold=1,
        data=None,
    ):
        original_length = len(all_extracted_periods)
        # print("Original length", original_length)
        i = 0
        while i < len(all_extracted_periods) - 1:
            if data.start_timestamp[
                all_extracted_periods[i + 1][0]
            ] - data.start_timestamp[
                all_extracted_periods[i][1]
            ] < merge_interval_threshold and not any(
                data.blink[
                    all_extracted_periods[i][1] : all_extracted_periods[i + 1][0]
                ]
            ):
                last_direction = data.gaze_direction[all_extracted_periods[i][1]]
                next_direction = data.gaze_direction[all_extracted_periods[i + 1][0]]
                if (
                    not utils.single_angular_distance(last_direction, next_direction)
                    > merge_direction_threshold
                ):
                    all_extracted_periods[i] = (
                        all_extracted_periods[i][0],
                        all_extracted_periods[i + 1][1],
                    )
                    all_extracted_periods.pop(i + 1)
                    continue

            i += 1
        # print("Merged", original_length - len(all_extracted_periods), "periods")
        return all_extracted_periods

    def update(self, data: GazeData) -> GazeData:
        fixations = []
        start = None
        all_periods = find_all_periods(data.fixation, indices=data.indices)
        data.fixation = np.zeros(len(data.fixation), dtype=bool)

        all_periods = self.merge_fixation_periods(
            all_periods,
            self.merge_interval_threshold,
            self.merge_direction_threshold,
            data,
        )
        all_periods = discard_short_periods(
            data, all_periods, self.short_fixation_threshold
        )

        for start, i in all_periods:
            flag = True
            data.fixation[start:i] = True

            # decide fixation target
            all_targets = data.label[start:i]
            # Determine the target based on threshold percentage
            # Only possible labels are 0 and 1

            if len(all_targets) > 0:
                # Count occurrences of label 1
                count_of_ones = np.sum(all_targets)
                percentage_of_ones = count_of_ones / len(all_targets)

                # If more than threshold% of targets are 1, set target to 1, otherwise 0
                if percentage_of_ones > self.target_threshold:
                    target = 1
                else:
                    target = 0
            else:
                target = None  # No targets available

            # eye_positions = data.eye_position[start:i]
            # gaze_directions = data.gaze_direction[start:i]
            # z_plane = -0.22
            # projected_2d_positions = np.array(
            #     utils.project_gazes_to_fix_plane(gaze_directions, eye_positions, -0.22)
            # )
            # _2d_center_position = np.average(projected_2d_positions, axis=0)
            # _2d_dispersion_x = np.max(projected_2d_positions[:, 0]) - np.min(
            #     projected_2d_positions[:, 0]
            # )
            # _2d_dispersion_y = np.max(projected_2d_positions[:, 1]) - np.min(
            #     projected_2d_positions[:, 1]
            # )
            # if _2d_dispersion_x > 0.1 or _2d_dispersion_y > 0.1:
            #     flag = False
            #     # try a different z-plane before moving forward
            #     for z_plane in [-0.21, -0.2, -0.19, -0.18, -0.17]:
            #         projected_2d_positions = np.array(
            #             utils.project_gazes_to_fix_plane(
            #                 gaze_directions, eye_positions, z_plane
            #             )
            #         )
            #         _2d_center_position = np.average(projected_2d_positions, axis=0)
            #         _2d_dispersion_x = np.max(projected_2d_positions[:, 0]) - np.min(
            #             projected_2d_positions[:, 0]
            #         )
            #         _2d_dispersion_y = np.max(projected_2d_positions[:, 1]) - np.min(
            #             projected_2d_positions[:, 1]
            #         )
            #         if _2d_dispersion_x < 0.1 and _2d_dispersion_y < 0.1:
            #             break
            #     if not flag:
            #         if target is None:
            #             _2d_center_position = [2, 2]
            #         else:
            #             _2d_center_position = [2, 0]
            # average_eye_position = np.mean(eye_positions, axis=0)
            # if flag:

            #     merged_3d_vergence_point = [
            #         _2d_center_position[0],
            #         _2d_center_position[1],
            #         z_plane,
            #     ]
            #     inferred_gaze_direction = (
            #         np.array(merged_3d_vergence_point) - average_eye_position
            #     )
            # else:
            #     inferred_gaze_direction = np.average(
            #         gaze_directions, axis=0, weights=data.duration[start:i]
            #     )
            #     # normalize the inferred gaze direction
            # inferred_gaze_direction = inferred_gaze_direction / np.linalg.norm(
            #     inferred_gaze_direction
            # )

            inferred_gaze_direction = np.average(
                data.gaze_direction[start:i], axis=0, weights=data.duration[start:i]
            )
            average_eye_position = np.average(
                data.eye_position[start:i], axis=0, weights=data.duration[start:i]
            )

            left_pupil_diameter = data.left_pupil_diameter[start:i]
            right_pupil_diameter = data.right_pupil_diameter[start:i]
            combined_pupil_diameter_mean = (
                np.mean(left_pupil_diameter) + np.mean(right_pupil_diameter)
            ) / 2

            self.total_fixation_duration += (
                data.start_timestamp[i] - data.start_timestamp[start]
            )

            fixations.append(
                {
                    "start_timestamp": data.start_timestamp[start],
                    "end_timestamp": data.start_timestamp[i],
                    "duration": data.start_timestamp[i] - data.start_timestamp[start],
                    # "2d_centroid": _2d_center_position,
                    # "2d_dispersion": (
                    #     _2d_dispersion_x + _2d_dispersion_y
                    #     if _2d_dispersion_x < 0.1 and _2d_dispersion_y < 0.1
                    #     else 0.5
                    # ),
                    # "z_plane": (
                    #     z_plane
                    #     if _2d_dispersion_x < 0.1 and _2d_dispersion_y < 0.1
                    #     else -1
                    # ),
                    "target": target,
                    "centroid": inferred_gaze_direction,
                    "eye_center": average_eye_position,
                    "dispersion": sum(
                        np.max(data.gaze_direction[start : i + 1], axis=0)
                        - np.min(data.gaze_direction[start : i + 1], axis=0)
                    ),
                    "start_index": start,
                    "end_index": i,
                    "mean_pupil": combined_pupil_diameter_mean,
                }
            )
            start = None

        data.fixations = fixations

        return data


class AggregateSaccades(Module):
    def update(self, data: GazeData) -> GazeData:
        saccades = []
        all_saccades = find_all_periods(data.saccade, indices=data.indices)
        # print("Number of saccades: ", len(all_saccades))
        for inner_start, inner_end in all_saccades:
            saccades.append(
                {
                    "start_timestamp": data.start_timestamp[inner_start],
                    "end_timestamp": data.start_timestamp[inner_end],
                    "start_index": inner_start,
                    "end_index": inner_end,
                    "duration": data.start_timestamp[inner_end]
                    - data.start_timestamp[inner_start],
                    "amplitude": utils.angular_distance(
                        data.gaze_direction[inner_start, np.newaxis],
                        data.gaze_direction[inner_end, np.newaxis],
                    ),
                    "velocity": np.mean(data.velocity[inner_start:inner_end]),
                    "peak_velocity": np.max(data.velocity[inner_start:inner_end]),
                    "source": (
                        data.label[inner_start - 1]
                        if inner_start > 0
                        else data.label[inner_start]
                    ),
                    "target": (
                        data.label[inner_end + 1]
                        if inner_end < len(data.label) - 1
                        else data.label[inner_end]
                    ),
                    "offset": (
                        data.gaze_direction[inner_end]
                        - data.gaze_direction[inner_start]
                    )
                    / (
                        data.start_timestamp[inner_end]
                        - data.start_timestamp[inner_start]
                    ),
                }
            )

        data.saccades = saccades
        # del data.saccade
        return data


class AggregateBlinks(Module):
    def __init__(self, min_count=2) -> None:
        super().__init__()
        self.min_count = min_count

    def update(self, data: GazeData) -> GazeData:
        reduced_total_duration = 0
        all_chunk_lengths = []
        blinks = []
        if hasattr(data, "blink_indices"):
            blink_indices = data.blink_indices
            # print("blink indices: ", blink_indices)
            count = 0
            chunks = []
            for i in range(len(blink_indices)):
                if i == 0 or blink_indices[i] != blink_indices[i - 1] + 1:
                    chunks.append([])
                chunks[-1].append(blink_indices[i])
            # merge chunks if they are too close, i.e. 1 frame apart
            for i in range(len(chunks) - 1):
                if chunks[i + 1][0] - chunks[i][-1] <= 1:
                    chunks[i] += chunks[i + 1]
                    chunks[i + 1] = []
            for chunk in chunks:
                if len(chunk) >= self.min_count:
                    if len(chunk) > 40:  # if too long, exclude it from the data
                        continue
                    start_index = np.where(data.indices == chunk[0] + data.first_index)[
                        0
                    ][0]
                    end_index = np.where(data.indices == chunk[-1] + data.first_index)[
                        0
                    ][0]
                    blinks.append(
                        {
                            "start_index": start_index,
                            "end_index": end_index + 1,
                            "duration": data.start_timestamp[end_index]
                            - data.start_timestamp[start_index],
                        }
                    )
        data.blinks = blinks
        return data


class AggregateSmoothPursuits(Module):
    def __init__(
        self,
        aggregate_to_fixations=False,
        short_sp_threshold=0.06,
        merge_interval_threshold=0.075,
    ) -> None:
        super().__init__()
        self.aggregate_to_fixations = aggregate_to_fixations
        self.short_sp_threshold = short_sp_threshold
        self.merge_interval_threshold = merge_interval_threshold

    def merge_sp_with_target_constraints(
        self,
        all_extracted_periods,
        merge_interval_threshold=0.075,
        merge_direction_threshold=0.5,
        data=None,
    ):
        original_length = len(all_extracted_periods)
        # print("Original length", original_length)
        i = 0
        while i < len(all_extracted_periods) - 1:
            if data.start_timestamp[
                all_extracted_periods[i + 1][0]
            ] - data.start_timestamp[
                all_extracted_periods[i][1]
            ] < merge_interval_threshold and not any(
                data.blink[
                    all_extracted_periods[i][1] : all_extracted_periods[i + 1][0]
                ]
            ):
                last_direction = data.gaze_direction[all_extracted_periods[i][1]]
                next_direction = data.gaze_direction[all_extracted_periods[i + 1][0]]
                if (
                    not utils.angular_distance([last_direction], [next_direction])[0]
                    > merge_direction_threshold
                ):
                    all_extracted_periods[i] = (
                        all_extracted_periods[i][0],
                        all_extracted_periods[i + 1][1],
                    )
                    all_extracted_periods.pop(i + 1)
                    continue
            i += 1
        # print("Merged", original_length - len(all_extracted_periods), "periods")
        return all_extracted_periods

    def update(self, data: GazeData) -> GazeData:
        smooth_pursuits = []
        all_smooth_pursuits = find_all_periods(
            data.smooth_pursuit, indices=data.indices
        )

        data.smooth_pursuit = np.zeros(len(data.smooth_pursuit), dtype=bool)
        all_smooth_pursuits = self.merge_sp_with_target_constraints(
            all_smooth_pursuits, data=data
        )
        all_smooth_pursuits = discard_short_periods(
            data, all_smooth_pursuits, self.short_sp_threshold
        )

        for start, i in all_smooth_pursuits:
            # print("Adding smooth pursuit: ", start, i)
            # print(data.fixation[start: i+1])
            # data.fixation[start: i+1] = True
            data.smooth_pursuit[start:i] = True

            # print("Smooth pursuit target: ", gaze_targets)
            # Get target mode if we have labels
            target_mode = None
            if hasattr(data, "label"):
                target_sequence = data.label[start:i]
                if len(target_sequence) > 0:
                    target_mode = mode(target_sequence, keepdims=True)[0][0]

            smooth_pursuits.append(
                {
                    "start_timestamp": data.start_timestamp[start],
                    "end_timestamp": data.start_timestamp[i],
                    "start_index": start,
                    "end_index": i,
                    "target": target_mode,
                    "duration": data.start_timestamp[i] - data.start_timestamp[start],
                    "amplitude": utils.angular_distance(
                        data.gaze_direction[start, np.newaxis],
                        data.gaze_direction[i, np.newaxis],
                    ),
                    "velocity": np.mean(data.velocity[start : i + 1]),
                }
            )
            if self.aggregate_to_fixations:
                data.fixations.append(
                    {
                        "start_timestamp": data.start_timestamp[start],
                        "end_timestamp": data.start_timestamp[i],
                        "duration": data.start_timestamp[i]
                        - data.start_timestamp[start],
                        "target": 1,
                    }
                )
        data.smooth_pursuits = smooth_pursuits
        return data


class GazeEventSequenceGenerator(Module):
    def update(self, data: GazeData) -> GazeData:
        # Create a numpy array filled with 'none' to represent no event
        event_sequence = np.full(len(data.start_timestamp), "none", dtype=object)

        # Fill in fixations
        for fixation in data.fixations:
            start_idx = fixation["start_index"]
            end_idx = fixation["end_index"]
            event_sequence[start_idx:end_idx] = "fixation"

        # Fill in saccades
        for saccade in data.saccades:
            start_idx = saccade["start_index"]
            end_idx = saccade["end_index"]
            event_sequence[start_idx:end_idx] = "saccade"

        # Fill in blinks (if available)
        if hasattr(data, "blinks"):
            for blink in data.blinks:
                start_idx = blink["start_index"]
                end_idx = blink["end_index"]
                event_sequence[start_idx:end_idx] = "blink"

        # Fill in smooth pursuits (if available)
        if hasattr(data, "smooth_pursuits"):
            for sp in data.smooth_pursuits:
                start_idx = sp["start_index"]
                end_idx = sp["end_index"]
                event_sequence[start_idx:end_idx] = "smooth_pursuit"

        # Add event sequence to data object
        data.event_sequence = event_sequence

        return data


class GazeDataExporter(Module):
    def __init__(
        self, export_path: str, metric_attr_map: dict = None, window_size=3
    ) -> None:
        self.export_path = export_path
        self.window_size = window_size
        if metric_attr_map is None:
            self.metric_attr_map = {
                "fixations": [
                    "duration",
                    "target",
                    "dispersion",
                    "centroid",
                    "start_index",
                    "end_index",
                    "2d_centroid",
                    "2d_dispersion",
                    "z_plane",
                    "eye_center",
                    "mean_pupil",
                ],
                "saccades": [
                    "duration",
                    "amplitude",
                    "offset",
                    "source",
                    "target",
                    "start_index",
                    "end_index",
                ],
                "blinks": ["duration", "start_index", "end_index"],
            }
        else:
            self.metric_attr_map = metric_attr_map

    def save_data(
        self, data: GazeData, user_id=1, trial="bleeding", detection=1
    ) -> GazeData:
        # export csv. Only save csv from [window_size //2, - window_size //2]
        df_copy = data.df.iloc[self.window_size // 2 : -self.window_size // 2][
            [
                "LeftPupilDiameter",
                "RightPupilDiameter",
                "EyeDirection_x",
                "EyeDirection_y",
                "EyeDirection_z",
                "EyePosition_x",
                "EyePosition_y",
                "EyePosition_z",
                "GazeTarget",
            ]
        ].copy()

        if hasattr(data, "event_sequence"):
            assert len(data.event_sequence) == len(df_copy)
            # First map event types to numeric values
            event_mapping = {
                "fixation": 1,
                "saccade": 2,
                "blink": 3,
            }

            # Convert string labels to numeric values
            numeric_events = np.array(
                [event_mapping.get(event, 0) for event in data.event_sequence]
            )

            # Create one-hot encoded columns
            df_copy["IsFixation"] = (numeric_events == 1).astype(int)
            df_copy["IsSaccade"] = (numeric_events == 2).astype(int)
            df_copy["IsBlink"] = (numeric_events == 3).astype(int)

            # Also keep the original categorical column
            # df_copy["GazeBehaviorType"] = data.event_sequence

        # if does not exist path, create
        if not os.path.exists(self.export_path):
            os.makedirs(self.export_path)

        df_copy.to_csv(
            f"{self.export_path}/{user_id}_{trial}_{detection}.csv", index=False
        )
        json_path = f"{self.export_path}/{user_id}_{trial}_{detection}.json"
        # export all gaze events to json.
        export_data = {
            "fixations": self.convert_metric_to_json(data, "fixations"),
            "saccades": self.convert_metric_to_json(data, "saccades"),
            "blinks": self.convert_metric_to_json(data, "blinks"),
        }
        # save json
        with open(json_path, "w") as f:
            json.dump(export_data, f, indent=4)

        return

    def convert_metric_to_json(self, data, metric):
        # data.metric is a list of dicts. For each dict, only keep the keys specified in metric_attr_map

        metric_list = getattr(data, f"{metric}", [])
        filtered_metric_list = []
        for item in metric_list:
            filtered_item = {}
            for key in self.metric_attr_map.get(metric, []):
                # Convert NumPy arrays to lists and handle other NumPy types
                if isinstance(item[key], np.ndarray):
                    filtered_item[key] = item[key].tolist()
                elif isinstance(item[key], np.number):
                    filtered_item[key] = item[key].item()
                else:
                    filtered_item[key] = item[key]
            filtered_metric_list.append(filtered_item)
        return filtered_metric_list


if __name__ == "__main__":
    a = np.array([[2, 3, 4], [-1, 4, 6], [9, 5, 5]])
