import tyro
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict, List, Union

Numeric = Union[float, int]

def compute_statistics(
    data: np.ndarray,
) -> Dict[str, List[Numeric]]:
    """Computes mean, std, q01, and q99 for the input data array.

    Args:
        data (np.ndarray): A numpy array (N, D) where N is the number of samples
                           and D is the dimension of the state/action space.

    Returns:
        Dict[str, List[Numeric]]: A dictionary containing the computed statistics.
                                  Keys are "mean", "std", "q01", "q99".
                                  Values are lists of numbers (one per dimension D).
    """
    stats = {
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
        "q01": np.quantile(data, 0.01, axis=0).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).tolist(),
    }
    return stats

def main(
    dataset_dir: Path,
    output_filename: str = "norm_stats.json",
    state_key: str = "joint_position",
    action_key: str = "actions",
) -> None:
    """
    Calculates normalization statistics (mean, std, q01, q99) for state
    and action data stored in parquet files within a LeRobot dataset structure.

    Args:
        dataset_dir (Path): The root directory of the LeRobot dataset.
                            Expected structure: dataset_dir/chunk-*/episode_*.parquet
        output_filename (str): The name of the output JSON file. Defaults to "norm_stats.json".
        state_key (str): The key in the parquet files corresponding to the state data.
                         Defaults to "joint_position".
        action_key (str): The key in the parquet files corresponding to the action data.
                          Defaults to "actions".
    """
    all_states = []
    all_actions = []

    print(f"Searching for parquet files in {dataset_dir}...")
    # Find all episode parquet files within chunk directories
    parquet_files = list(dataset_dir.rglob("chunk-*/episode_*.parquet"))

    if not parquet_files:
        print(f"Error: No parquet files found in the expected structure under {dataset_dir}")
        return

    print(f"Found {len(parquet_files)} parquet files. Reading data...")
    for file_path in tqdm(parquet_files, desc="Reading parquet files"):
        try:
            df = pd.read_parquet(file_path, columns=[state_key, action_key])
            # Assuming data is stored as lists/arrays within cells, stack them
            states = np.vstack(df[state_key].values)
            actions = np.vstack(df[action_key].values)
            all_states.append(states)
            all_actions.append(actions)
        except Exception as e:
            print(f"Warning: Could not read or process file {file_path}. Error: {e}")

    if not all_states or not all_actions:
        print("Error: No valid state or action data could be extracted.")
        return

    print("Concatenating data...")
    # Concatenate all data into large numpy arrays
    all_states_np = np.concatenate(all_states, axis=0)
    all_actions_np = np.concatenate(all_actions, axis=0)

    print(f"State data shape: {all_states_np.shape}")
    print(f"Action data shape: {all_actions_np.shape}")

    print("Computing statistics...")
    # Compute statistics
    state_stats = compute_statistics(all_states_np)
    action_stats = compute_statistics(all_actions_np)

    # Structure the final output
    output_data = {
        "norm_stats": {
            "state": state_stats,
            "actions": action_stats,
        }
    }

    # Save the results
    output_path = dataset_dir / output_filename
    print(f"Saving statistics to {output_path} ...")
    try:
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print("Successfully saved statistics.")
    except Exception as e:
        print(f"Error saving statistics file: {e}")

if __name__ == "__main__":
    tyro.cli(main)
