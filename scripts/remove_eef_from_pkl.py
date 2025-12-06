import json
import os
import shutil
from dataclasses import dataclass
from glob import glob

import numpy as np
import pandas as pd
import tyro
from tqdm import tqdm

pd.options.mode.chained_assignment = None


def remove_eef_from_parquet(parquet_path: str) -> pd.DataFrame:
    """Load a parquet file and drop the first 32 dims from state/action vectors."""
    df = pd.read_parquet(parquet_path)

    for column in ("observation.state", "action"):
        stacked = np.stack(df[column].to_numpy())
        # reorder the stacked columns 
        left_gripper = stacked[:, 32:33]
        right_gripper = stacked[:, 33:34]
        left_joints = stacked[:, 34:40]
        right_joints = stacked[:, 40:46]
        new_order = np.concatenate([left_joints, left_gripper, right_joints, right_gripper], axis=1)
        df[column] = list(new_order)  # keep per-row array objects

    return df

def process_folder(folder_path: str) -> None:
    """Apply the EEF removal transform to every parquet file in folder_path/data."""
    data_dir = os.path.join(folder_path, "data")
    output_dir = os.path.join(folder_path, "data_no_eef")
    os.makedirs(output_dir, exist_ok=True)

    for parquet_path in tqdm(glob(os.path.join(data_dir, "**/*.parquet"), recursive=True)):
        parent_folder = os.path.dirname(parquet_path)
        output_chunk_dir = parent_folder.replace(data_dir, output_dir)
        os.makedirs(output_chunk_dir, exist_ok=True)

        df = remove_eef_from_parquet(parquet_path)
        df.to_parquet(parquet_path.replace(data_dir, output_dir))

    # rename data_dir to data_old 
    os.rename(data_dir, os.path.join(folder_path, "data_with_eef"))
    os.rename(output_dir, data_dir)

    # stats path 
    stats_path = os.path.join(folder_path, "meta", "stats.json")
    # copy stats to stats_no_eef.json
    shutil.copy(stats_path, os.path.join(folder_path, "meta", "stats_with_eef.json"))

    with open(stats_path, "r") as f:
        stats = json.load(f)
    for key in ["observation.state", "action"]:
        for stat_type in stats[key]:
            left_gripper = stats[key][stat_type][32:33]
            right_gripper = stats[key][stat_type][33:34]
            left_joints = stats[key][stat_type][34:40]
            right_joints = stats[key][stat_type][40:46]
            new_order = left_joints + left_gripper + right_joints + right_gripper
            stats[key][stat_type] = new_order
    with open(stats_path, "w") as f:
        json.dump(stats, f)

@dataclass
class Args:
    """CLI arguments."""

    folder_path: str


def main() -> None:
    args = tyro.cli(Args)
    process_folder(args.folder_path)


if __name__ == "__main__":
    main()