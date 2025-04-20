"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import os 
# os.environ["LEROBOT_HOME"] = "/mnt/disks/ssd1/lerobot"
os.environ["LEROBOT_HOME"] = "/shared/projects/icrl/data/dpgs/lerobot"
import shutil
import h5py 
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm, trange
import zarr
from PIL import Image
from openpi_client.image_tools import resize_with_pad
import matplotlib.pyplot as plt
import tyro
from typing import List, Tuple

RAW_DATASET_FOLDERS = [
    # "/mnt/disks/ssd7/dpgs_dataset/yumi_coffee_maker/successes_041325"
    # "/mnt/disks/ssd7/dpgs_dataset/yumi_faucet/successes_041425",
    # "/mnt/disks/ssd7/dpgs_dataset/yumi_led_light/successes_041425_2334"
    # "/shared/projects/dpgs_dataset/yumi_bin_pickup/successes_041625_2054",
    "/shared/projects/dpgs_dataset/yumi_drawer_open/successes_041725_2136/successes",
    # "/shared/projects/dpgs_dataset/yumi_pick_tiger_r2r2r/successes_041725_2203",
    # "/shared/projects/dpgs_dataset/yumi_led_light/successes_041825_1856",
    # "/shared/projects/dpgs_dataset/yumi_cardboard_lift/successes_041825_2245",
    # "/shared/projects/dpgs_dataset/yumi_faucet/successes_041425" # bajcsy
    # "/shared/projects/dpgs_dataset/yumi_drawer_open/successes_041525_2044" # bajcsy
]
LANGUAGE_INSTRUCTIONS = [
    # # "put the white cup on the coffee machine"
    "open the drawer"
    # "pick up the tiger"
    # "turn off the faucet"
    # "turn the LED light"
    # "pick up the cardboard box"
    # "pick up the bin"
]

# # REPO_NAME = "mlfu7/dpgs_sim_faucet_maker_5k_updated"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_sim_faucet_5k"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_sim_drawer_open_1k"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_sim_led_5k"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_sim_bin_pickup_1k"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_sim_drawer_open_new_1k"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_sim_tiger_1k"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_sim_led_v2_1k"  # Name of the output dataset, also used for the Hugging Face Hub
# REPO_NAME = "mlfu7/dpgs_bimanual_lift_v2_1k"  # Name of the output dataset, also used for the Hugging Face Hub

CAMERA_KEYS = [
    "camera_0/rgb", 
    "camera_1/rgb"
] # folder of rgb images
CAMERA_KEY_MAPPING = {
    "camera_0/rgb": "exterior_image_1_left",
    "camera_1/rgb": "exterior_image_2_left",
}
STATE_KEY = "robot_data/robot_data_joint.zarr"
RESIZE_SIZE = 224

def calculate_individual_joint_deltas(data_dirs: List[str], state_key: str) -> List[float]:
    """
    Calculates the absolute delta for each individual joint across all timesteps and trajectories.

    Args:
        data_dirs (List[str]): List of directories containing trajectory data.
        state_key (str): The relative path within each trajectory folder to the zarr file containing proprioceptive data.

    Returns:
        List[float]: A flattened list containing the absolute delta (abs(pos[t+1] - pos[t])) for every joint at every timestep.
    """
    all_individual_deltas = []
    for data_day_dir in data_dirs:
        print(f"Processing folder: {data_day_dir}")
        if not os.path.isdir(data_day_dir):
            print(f"Warning: Directory not found: {data_day_dir}")
            continue
        try:
            trajs = os.listdir(data_day_dir)
        except FileNotFoundError:
            print(f"Error: Cannot access directory: {data_day_dir}")
            continue

        for idx, task in enumerate(tqdm(trajs, desc=f"Processing {os.path.basename(data_day_dir)}")):
            task_folder = os.path.join(data_day_dir, task)
            proprio_file_path = os.path.join(task_folder, state_key)

            if not os.path.exists(proprio_file_path):
                print(f"Warning: Proprio file not found: {proprio_file_path}")
                continue

            try:
                proprio_data = zarr.load(proprio_file_path)
            except Exception as e:
                print(f"Error loading zarr file {proprio_file_path}: {e}")
                continue

            if proprio_data.shape[0] <= 1:
                print(f"Warning: Insufficient data in {proprio_file_path}")
                continue

            # Calculate deltas for all timesteps
            # proprio_data shape: (num_timesteps, num_joints)
            # deltas shape: (num_timesteps - 1, num_joints)
            deltas = np.diff(proprio_data, axis=0)
            abs_deltas = np.abs(deltas)

            # Flatten the array of absolute deltas and extend the main list
            all_individual_deltas.extend(abs_deltas.flatten())

    return all_individual_deltas

def plot_histogram(deltas: List[float], bins: int = 100, title: str = "Distribution of Absolute Per-Joint Deltas", xlabel: str = "Absolute Delta per Joint", ylabel: str = "Frequency", save_path: str | None = "joint_delta_histogram.png"):
    """
    Plots a histogram of the given individual absolute joint deltas.

    Args:
        deltas (List[float]): Flattened list of absolute per-joint deltas.
        bins (int): Number of bins for the histogram.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        save_path (str | None): Path to save the plot. If None, the plot is displayed instead.
    """
    if not deltas:
        print("No joint deltas to plot.")
        return

    plt.figure(figsize=(12, 7))
    # Use numpy histogram to get counts and bins, especially for log scale plotting
    counts, bin_edges = np.histogram(deltas, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.bar(bin_centers, counts, width=np.diff(bin_edges)[0]) # Use bar plot for better control with log scale

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale('log') # Use log scale for better visibility of small deltas
    plt.grid(True, which="both", ls="--")

    # Calculate and print some statistics
    mean_delta = np.mean(deltas)
    median_delta = np.median(deltas)
    std_delta = np.std(deltas)
    min_delta = np.min(deltas)
    max_delta = np.max(deltas)
    percentile_1 = np.percentile(deltas, 1)
    percentile_5 = np.percentile(deltas, 5)
    percentile_10 = np.percentile(deltas, 10)
    percentile_90 = np.percentile(deltas, 90)
    percentile_95 = np.percentile(deltas, 95)
    percentile_99 = np.percentile(deltas, 99)


    stats_text = (
        f"Mean: {mean_delta:.6f}\n"
        f"Median: {median_delta:.6f}\n"
        f"Std Dev: {std_delta:.6f}\n"
        f"Min: {min_delta:.6f}\n"
        f"Max: {max_delta:.6f}\n"
        f"1st Pct: {percentile_1:.6f}\n"
        f"5th Pct: {percentile_5:.6f}\n"
        f"10th Pct: {percentile_10:.6f}\n"
        f"90th Pct: {percentile_90:.6f}\n"
        f"95th Pct: {percentile_95:.6f}\n"
        f"99th Pct: {percentile_99:.6f}"
    )
    print("\nAbsolute Per-Joint Delta Statistics:")
    print(stats_text)

    # Add stats text to the plot
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


    if save_path:
        plt.savefig(save_path)
        print(f"Histogram saved to {save_path}")
    else:
        plt.show()

def main(
    data_dirs: List[str] = RAW_DATASET_FOLDERS,
    state_key: str = STATE_KEY,
    plot_bins: int = 100,
    output_file: str = "joint_delta_histogram.png"
    ):
    """
    Main function to calculate individual absolute joint deltas and plot their distribution.

    Args:
        data_dirs (List[str]): List of directories containing trajectory data. Defaults to RAW_DATASET_FOLDERS.
        state_key (str): Relative path to the zarr proprioceptive data file. Defaults to STATE_KEY.
        plot_bins (int): Number of bins for the histogram plot. Defaults to 100.
        output_file (str): Path to save the output histogram plot. Defaults to "joint_delta_histogram.png".
    """
    print("Calculating individual joint deltas...")
    deltas = calculate_individual_joint_deltas(data_dirs, state_key)
    print(f"Calculated {len(deltas)} individual joint delta values.")

    if deltas:
        print("Plotting histogram...")
        plot_histogram(deltas, bins=plot_bins, save_path=output_file)
    else:
        print("No data found to plot.")


if __name__ == "__main__":
    # Use tyro to parse command-line arguments
    tyro.cli(main)
