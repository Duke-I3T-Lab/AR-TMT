import numpy as np
from typing import Optional, Dict

from offline.data import GazeData
from offline.modules import Module


class IVTFixationDetector(Module):
    def __init__(
        self,
        velocity_threshold: float = 30,
        dynamic_threshold=False,
        use_mobility=True,
        mobile_velocity_threshold: Optional[float] = None,
    ) -> None:
        self.velocity_threshold = velocity_threshold
        self.dynamic_threshold = dynamic_threshold
        self.use_mobility = use_mobility
        self.mobile_velocity_threshold = mobile_velocity_threshold

    def update(self, data: GazeData) -> GazeData:
        data.fixation = data.velocity < self.velocity_threshold
        # for i in range(len(data.fixation)):
        #     if 500 < i < 600:
        #         print("i: ", i)
        #         print("timestamp: ", data.start_timestamp[i])
        #         print("velocity: ", data.velocity[i])
        #         print("delta: ", data.delta[i])
        #         print("=" * 20)
        # exit()
        if self.dynamic_threshold:
            # print(
            #     np.sum(data.delta < 0), " negative deltas found in dynamic thresholding"
            # )
            for i in range(len(data.velocity)):
                if (
                    not data.fixation[i]
                    and data.velocity[i]
                    < data.velocity_threshold_max[i] + self.velocity_threshold
                    and data.delta[i] < 1e-5
                ):
                    print("i: ", i)
                    print("Time Stamp: ", data.start_timestamp[i])
                    print("Velocity: ", data.velocity[i])
                    # print("upper threshold: ", data.velocity_threshold_max)
                    print(
                        "modified upper threshold: ",
                        data.velocity_threshold_max[i] + self.velocity_threshold,
                    )

            # data.fixation = np.logical_or(data.fixation, np.logical_and(data.velocity < data.velocity_threshold_max, data.velocity > data.velocity_threshold_min))
            data.fixation = np.logical_or(
                data.fixation,
                np.logical_and(
                    data.delta < 1e-4,
                    data.velocity
                    < data.velocity_threshold_max + self.velocity_threshold,
                ),
            )
            # count number of times fixation is true
            # print("count: ", np.count_nonzero(data.fixation))
            # print("length: ", len(data.fixation))
        if self.use_mobility:
            data.fixation = np.logical_or(
                data.fixation,
                np.logical_and(
                    data.mobility_mask, data.velocity < self.mobile_velocity_threshold
                ),
            )
        if hasattr(data, "blink"):
            data.fixation = np.logical_and(data.fixation, ~data.blink)

        return data


class BlinkConverter(Module):
    def __init__(self) -> None:
        pass

    def update(self, data: GazeData) -> GazeData:
        # convert blink indices to a numpy list of booleans
        data.blink = np.zeros(len(data), dtype=bool)
        data.blink[data.blink_indices] = True
        return data


class IVTSaccadeDetector(Module):
    def __init__(self, velocity_threshold: float = 30) -> None:
        self.velocity_threshold = velocity_threshold

    def update(self, data: GazeData) -> GazeData:
        # saccade should be neg of fixation
        # data.saccade = data.velocity > self.velocity_threshold
        data.saccade = ~data.fixation
        if hasattr(data, "blink"):
            data.saccade = np.logical_and(data.saccade, ~data.blink)
        return data


class SmoothPursuitDetector(Module):
    def __init__(
        self,
        low_velocity_threshold: float = 10,
        high_velocity_threshold: float = 40,
        dynamic_threshold=False,
    ) -> None:
        self.low_velocity_threshold = low_velocity_threshold
        self.high_velocity_threshold = high_velocity_threshold
        self.dynamic_threshold = dynamic_threshold

    def update(self, data: GazeData) -> GazeData:
        data.smooth_pursuit = (data.velocity > self.low_velocity_threshold) & (
            data.velocity < self.high_velocity_threshold
        )
        if self.dynamic_threshold:
            # data.smooth_pursuit = np.logical_or(data.smooth_pursuit, np.logical_and(data.velocity < data.velocity_threshold_max + self.high_velocity_threshold - self.low_velocity_threshold, data.velocity > data.velocity_threshold_min + self.low_velocity_threshold - self.low_velocity_threshold))
            data.smooth_pursuit = np.logical_or(
                data.smooth_pursuit,
                np.logical_and(
                    data.velocity
                    < data.velocity_threshold_max + self.high_velocity_threshold,
                    data.velocity
                    > self.low_velocity_threshold + data.velocity_threshold_min,
                ),
            )

        return data
