# filepath: d:\Research\I3T\Projects\Sihun_ARTMT\src\gaze_extraction\artmt_main.py
import copy
import numpy as np
import pandas as pd
import os
import json

import offline.modules as m
from offline.data import ARTMTGazeData, CPREyeTrackingMetric
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "data"
RESULTS_DIR = "src/gaze_extraction/results/artmt"

# Trial type mapping based on task numbers
TRIAL_TYPE_MAPPING = {
    1: "TMT-A",
    2: "TMT-B",
    3: "Neutral",
    4: "top-down",
    5: "bottom-up",
    6: "spatial",
}

# Available participants (1-19 based on directory listing)
PARTICIPANTS = list(range(1, 20))

# Metrics to extract - focusing on the required ones: MFD, FR, PFT, MSV, MSA, MPSV
METRICS = {
    "fixation_metrics": ["MFD", "FR", "PFT"],
    "saccade_metrics": ["MSV", "MPSV", "MSA"],
    "blink_metrics": ["BR"],
    "diameter_metrics": ["MD"],
}

NORMALIZE_PUPIL_DIAMETER = True


def load_acs_data(acs_file_path):
    """Load ACS summary data and return as dictionary with UserID as key"""
    acs_df = pd.read_csv(acs_file_path)
    acs_dict = {}
    for _, row in acs_df.iterrows():
        user_id = row["User"]
        acs_before = row["ACS before"]
        acs_dict[user_id] = acs_before
    return acs_dict


def write_per_user_data(
    user_data,
    summarized_df=None,
    user_id=1,
    ACS=28,
    motor_time_avg=1,
    trial_type="TMT-A",
    completion_time=0,
    distractor_hits=0,
    miss_hits=0,
    wrong_hits=0,
):
    """Write per-user data to DataFrame after each file is processed"""
    if summarized_df is None:
        # Initialize DataFrame with required columns
        metric_columns = []
        for metric_names in METRICS.values():
            metric_columns.extend(metric_names)

        columns = [
            "UserID",
            "ACS",
            "Trial_Type",
            "MotorSkill",
            "CompletionTime",
            "DistractorHits",
            "MissHits",
            "WrongHits",
        ] + metric_columns
        summarized_df = pd.DataFrame(columns=columns)

    # Extract metrics from user_data and create a row
    row_data = {
        "UserID": user_id,
        "ACS": ACS,
        "MotorSkill": motor_time_avg,
        "Trial_Type": trial_type,
        "CompletionTime": completion_time,
        "DistractorHits": distractor_hits,
        "MissHits": miss_hits,
        "WrongHits": wrong_hits,
    }

    for event, metric_classes in METRICS.items():
        for metric_class in metric_classes:
            if hasattr(user_data, event):
                metric_value = getattr(user_data, event).get(metric_class)
                if metric_value is not None:
                    row_data[metric_class] = metric_value

    # Append the row to the DataFrame
    new_row = pd.DataFrame([row_data])
    summarized_df = pd.concat([summarized_df, new_row], ignore_index=True)

    return summarized_df


if __name__ == "__main__":

    # Set up processing modules (same as CPR but without ROI-specific modules)
    modules = [
        m.BlinkConverter(),
        m.DurationDistanceVelocity(window_size=3),
        # m.MobilityDetection(window_size=5),
        m.SavgolFilter(attr="velocity", window_size=3, order=1),
        m.IVTFixationDetector(
            velocity_threshold=30,
            dynamic_threshold=False,
            use_mobility=False,
            # mobile_velocity_threshold=100,
        ),
        m.AggregateFixations(merge_direction_threshold=0.5, target_threshold=0.5),
        m.IVTSaccadeDetector(velocity_threshold=30),
        m.AggregateSaccades(),
        m.AggregateBlinks(),
        # m.GazeEventSequenceGenerator(),
        m.FixationMetrics(),
        m.SaccadeMetrics(),
        m.BlinkMetrics(min_count=2),
        m.DiameterMetrics(),
    ]

    # Load ACS data
    acs_file_path = os.path.join(DATA_DIR, "ACS_summary.csv")
    acs_data = load_acs_data(acs_file_path)
    print(f"Loaded ACS data for {len(acs_data)} participants")

    results_df = None

    results = {}
    processed_count = 0

    LEFT_PUPIL_DIAMETER_NORMALIZER, RIGHT_PUPIL_DIAMETER_NORMALIZER = (
        StandardScaler(),
        StandardScaler(),
    )

    print("Processing ARTMT eyetracking data...")
    for participant in tqdm(PARTICIPANTS):
        p_data_dir = os.path.join(DATA_DIR, str(participant))

        if not os.path.exists(p_data_dir):
            print(f"Warning: Data directory not found for participant {participant}")
            continue

        # read motor speed data
        motor_speed_file = os.path.join(p_data_dir, "Performancedata_task0.json")
        with open(motor_speed_file, "r") as f:
            motor_speed_data = json.load(f)
            motor_time_list = motor_speed_data.get("Completion_time_list", [0, 1])
            motor_time_avg = np.mean(np.array(motor_time_list[1:]))

        # Process each task (1-6)
        for task_num in range(1, 7):
            print("Processing user:", participant, "task:", task_num)
            eyetracking_file = os.path.join(
                p_data_dir, f"eyetracking_task{task_num}.json"
            )
            performance_file = os.path.join(
                p_data_dir, f"Performancedata_task{task_num}.json"
            )
            # read performance data, json file
            if not os.path.exists(performance_file):
                print(f"Warning: Performance file not found: {performance_file}")
                continue
            with open(performance_file, "r") as f:
                performance_data = json.load(f)
                completion_time = performance_data.get("CompletionTime", 0)
                distractor_hit_count = performance_data.get(
                    "NumberOfHittingDistractors", 0
                )
                miss_hit_count = performance_data.get("NumberOfMissHits", 0)
                wrong_hit_count = performance_data.get("NumberOfWrongOrderHits", 0)

            if not os.path.exists(eyetracking_file):
                print(f"Warning: Eyetracking file not found: {eyetracking_file}")
                continue

            # Load ARTMT gaze data
            gaze_data = ARTMTGazeData(eyetracking_file)

            if task_num == 1:
                left_pupil_baseline = gaze_data.left_pupil_diameter
                right_pupil_baseline = gaze_data.right_pupil_diameter
                LEFT_PUPIL_DIAMETER_NORMALIZER.fit(left_pupil_baseline.reshape(-1, 1))
                RIGHT_PUPIL_DIAMETER_NORMALIZER.fit(right_pupil_baseline.reshape(-1, 1))

            if NORMALIZE_PUPIL_DIAMETER:
                gaze_data.normalize_pupil_diameter(
                    LEFT_PUPIL_DIAMETER_NORMALIZER,
                    RIGHT_PUPIL_DIAMETER_NORMALIZER,
                    use_normal_dist=True,
                )

            # Skip if no valid data
            if len(gaze_data.gaze_direction) == 0:
                print(
                    f"Warning: No valid gaze data for participant {participant}, task {task_num}"
                )
                continue

            # Process data through the pipeline
            processed_data = copy.deepcopy(gaze_data)
            for module in modules:
                processed_data = module.update(processed_data)

            # Get trial type
            trial_type = TRIAL_TYPE_MAPPING[task_num]
            print(f"Trial type: {trial_type}")

            results_df = write_per_user_data(
                processed_data,
                summarized_df=results_df,
                user_id=participant,
                ACS=acs_data.get(participant, 0),
                motor_time_avg=motor_time_avg,
                trial_type=trial_type,
                completion_time=completion_time,
                distractor_hits=distractor_hit_count,
                miss_hits=miss_hit_count,
                wrong_hits=wrong_hit_count,
            )

            processed_count += 1

    print(f"Successfully processed {processed_count} files")

    if results_df.empty:
        print("No results to save!")
        exit(1)

    # Sort by UserID and Trial Type for better readability
    final_df = results_df.sort_values(["UserID", "Trial_Type"]).reset_index(drop=True)

    # Save results
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    output_file = os.path.join(RESULTS_DIR, "artmt_gaze_metrics.csv")
    final_df.to_csv(output_file, index=False)

    print(f"Results saved to: {output_file}")
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Columns: {list(final_df.columns)}")
    print(f"Participants processed: {sorted(final_df['UserID'].unique())}")
    print(f"Trial types: {sorted(final_df['Trial_Type'].unique())}")

    # Display sample of results
    print("\nSample of results:")
    print(final_df.head(10))
