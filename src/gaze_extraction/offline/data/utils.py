import copy
import numpy as np
from typing import List

from data import GazeData

def extract_all_windows_from_labeled_data(data_list: List[GazeData], window_size=120, step_size=120):
    windows = []
    for data in data_list:
        # now we still have to deal inconsistency in index vs. df index
        full_length = len(data)
        start = 0
        while start < full_length - window_size + 1:
            end = start + window_size
            while data.indices[end-1] - data.indices[start] != window_size - 1 and end > start + 1:
                end -= 1
            if end >= start + 60:
                windows.append(copy.deepcopy(data).slice_index(start, end))
            start = end
    return windows

