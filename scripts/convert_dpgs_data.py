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
import tyro
from typing import List, Dict

# Default values, can be overridden by CLI args
DEFAULT_RAW_DATASET_FOLDERS = [
    # "/mnt/disks/ssd7/dpgs_dataset/yumi_coffee_maker/successes_041325"
    # "/mnt/disks/ssd7/dpgs_dataset/yumi_faucet/successes_041425",
    # "/mnt/disks/ssd7/dpgs_dataset/yumi_led_light/successes_041425_2334"
    # "/shared/projects/dpgs_dataset/yumi_bin_pickup/successes_041625_2054",
    # "/shared/projects/dpgs_dataset/yumi_drawer_open/successes_041725_2136/successes",
    # "/shared/projects/dpgs_dataset/yumi_pick_tiger_r2r2r/successes_041725_2203",
    # "/shared/projects/dpgs_dataset/yumi_led_light/successes_041825_1856",
    # # "/shared/projects/dpgs_dataset/yumi_cardboard_lift/successes_041825_2245",
    # "/shared/projects/dpgs_dataset/yumi_faucet/successes_041425" # bajcsy
    "/shared/projects/dpgs_dataset/yumi_drawer_open/successes_042025_1743",
    # "/shared/projects/dpgs_dataset/yumi_drawer_open/successes_041525_2044" # bajcsy
    # "/shared/projects/dpgs_dataset/yumi_drawer_open/successes_041925_2005/successes_041925_2005",
    
]
DEFAULT_LANGUAGE_INSTRUCTIONS = [
    # # "put the white cup on the coffee machine"
    "open the drawer"
    # "pick up the tiger"
    # "turn off the faucet"
    # "turn the LED light"
    # "pick up the cardboard box"
    # "pick up the bin"
]
# DEFAULT_REPO_NAME = "mlfu7/dpgs_sim_drawer_open_10_test"  # Default output dataset name
# DEFAULT_REPO_NAME = "mlfu7/dpgs_sim_tiger_1k_v2"  # Default output dataset name
# DEFAULT_REPO_NAME = "mlfu7/dpgs_sim_drawer_open_1k_v3"  # Default output dataset name
DEFAULT_REPO_NAME = "mlfu7/dpgs_sim_drawer_open_1k_v4"  # Default output dataset name
# DEFAULT_REPO_NAME = "mlfu7/dpgs_sim_faucet_1k"  # Default output dataset name

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

def main(
    raw_dataset_folders: List[str] = DEFAULT_RAW_DATASET_FOLDERS,
    language_instructions: List[str] = DEFAULT_LANGUAGE_INSTRUCTIONS,
    repo_name: str = DEFAULT_REPO_NAME,
    delta_threshold: float = 1e-5,
    camera_keys: List[str] = CAMERA_KEYS,
    camera_key_mapping: Dict[str, str] = CAMERA_KEY_MAPPING,
    state_key: str = STATE_KEY,
    resize_size: int = RESIZE_SIZE,
    fps: int = 15,
    upto_n_traj: int = None,
    robot_type: str = "yumi", # Or "yumi" etc.
    image_writer_threads: int = 20,
    image_writer_processes: int = 10,
    ):
    """
    Converts raw trajectory data (proprio, images) into LeRobot format,
    filtering out timesteps where joint movement is below a threshold.

    Args:
        raw_dataset_folders: List of paths to raw dataset folders.
        language_instructions: List of language instructions corresponding to each folder.
        repo_name: Name for the output LeRobot dataset (and Hugging Face Hub repo).
        delta_threshold: Minimum absolute delta required in at least one joint to keep the timestep.
        camera_keys: List of camera folder names in the raw data.
        camera_key_mapping: Dictionary mapping raw camera keys to LeRobot feature keys.
        state_key: Relative path to the proprioceptive data file (e.g., zarr) within each trajectory folder.
        resize_size: Target size (height and width) for resizing images.
        fps: Frames per second for the dataset.
        upto_n_traj: Number of trajectories to process.
        robot_type: Type of robot used (e.g., 'panda', 'yumi').
        image_writer_threads: Number of threads for image writing.
        image_writer_processes: Number of processes for image writing.
    """
    if len(raw_dataset_folders) != len(language_instructions):
        raise ValueError("Number of dataset folders must match number of language instructions.")

    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / repo_name
    if output_path.exists():
        print(f"Warning: Existing dataset found at {output_path}. Removing it.")
        shutil.rmtree(output_path)
    print("Dataset will be saved to:", output_path)

    # --- Create LeRobot dataset ---
    # Assuming 16 joints based on previous example, adjust if needed
    # Infer shape from first valid trajectory? Better to be explicit or check.
    # Let's keep 16 for now.
    joint_dim = 16
    try:
        # Attempt to infer joint dim from the first valid file
        found_dim = False
        for folder in raw_dataset_folders:
             if not os.path.isdir(folder): continue
             trajs = os.listdir(folder)
             if not trajs: continue
             first_traj_path = os.path.join(folder, trajs[0], state_key)
             if os.path.exists(first_traj_path):
                 try:
                     proprio_sample = zarr.load(first_traj_path)
                     if proprio_sample.ndim >= 2 and proprio_sample.shape[1] > 0:
                         joint_dim = proprio_sample.shape[1]
                         print(f"Inferred joint dimension: {joint_dim}")
                         found_dim = True
                         break
                 except Exception as e:
                     print(f"Could not read {first_traj_path} to infer dim: {e}")
             if found_dim: break
        if not found_dim:
             print(f"Warning: Could not infer joint dimension, defaulting to {joint_dim}")
    except Exception as e:
        print(f"Error during joint dimension inference: {e}. Defaulting to {joint_dim}")


    image_features = {
        key_map: {
            "dtype": "video",
            "shape": (resize_size, resize_size, 3),
            "names": ["height", "width", "channel"],
        } for key_map in camera_key_mapping.values()
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type=robot_type,
        fps=fps,
        features={
            **image_features,
            "joint_position": {
                "dtype": "float32",
                "shape": (joint_dim,),
                "names": ["joint_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (joint_dim,),
                "names": ["actions"],
            },
        },
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    total_frames_processed = 0
    total_frames_added = 0
    total_episodes_saved = 0

    # Loop over raw datasets
    for raw_dataset_name, language_instruction in zip(raw_dataset_folders, language_instructions):
        data_day_dir = raw_dataset_name
        if not os.path.isdir(data_day_dir):
            print(f"Warning: Skipping non-existent directory {data_day_dir}")
            continue
        print(f"Processing folder: {data_day_dir}")
        try:
            trajs = os.listdir(data_day_dir)
        except OSError as e:
            print(f"Error listing directory {data_day_dir}: {e}. Skipping.")
            continue

        if upto_n_traj is not None:
            print(f"Processing only first {upto_n_traj} trajectories")
            trajs = trajs[:upto_n_traj]

        for idx, task in enumerate(tqdm(trajs, desc=f"Processing {os.path.basename(data_day_dir)}")):
            task_folder = os.path.join(data_day_dir, task)
            proprio_file_path = os.path.join(task_folder, state_key)

            if not os.path.exists(proprio_file_path):
                print(f"Warning: Proprio file not found: {proprio_file_path}. Skipping trajectory {task}.")
                continue

            try:
                proprio_data = zarr.load(proprio_file_path)
            except Exception as e:
                print(f"Error loading zarr file {proprio_file_path}: {e}. Skipping trajectory {task}.")
                continue

            if proprio_data.shape[0] <= 1:
                print(f"Warning: Insufficient data points ({proprio_data.shape[0]}) in {proprio_file_path}. Skipping trajectory {task}.")
                continue

            if proprio_data.shape[1] != joint_dim:
                 print(f"Warning: Unexpected joint dimension ({proprio_data.shape[1]}, expected {joint_dim}) in {proprio_file_path}. Skipping trajectory {task}.")
                 continue


            # Check if image folders exist and have enough images
            image_paths_dict = {}
            min_image_count = float('inf')
            valid_images = True
            for raw_cam_key in camera_keys:
                cam_folder = os.path.join(task_folder, raw_cam_key)
                if not os.path.isdir(cam_folder):
                    print(f"Warning: Image directory not found: {cam_folder}. Skipping trajectory {task}.")
                    valid_images = False
                    break
                try:
                    image_files = sorted(os.listdir(cam_folder))
                    image_paths_dict[raw_cam_key] = [os.path.join(cam_folder, f) for f in image_files]
                    min_image_count = min(min_image_count, len(image_files))
                except OSError as e:
                    print(f"Error listing image directory {cam_folder}: {e}. Skipping trajectory {task}.")
                    valid_images = False
                    break
            if not valid_images:
                continue

            # sequence length is determined by proprio data - 1 for action calculation
            seq_length = proprio_data.shape[0] - 1
            if min_image_count < seq_length:
                 print(f"Warning: Insufficient images ({min_image_count}) for sequence length ({seq_length}) in {task_folder}. Skipping trajectory {task}.")
                 continue

            last_added_proprio = None
            num_frames_added_episode = 0

            for step in range(seq_length):
                total_frames_processed += 1
                proprio_t = proprio_data[step]
                proprio_t_plus_1 = proprio_data[step + 1] # Needed for action calculation

                should_add_frame = True
                if last_added_proprio is not None:
                    # Calculate difference from the last *added* state
                    proprio_delta = np.abs(proprio_t - last_added_proprio)
                    # If *all* joint deltas are below the threshold, skip
                    if np.all(proprio_delta < delta_threshold):
                        should_add_frame = False

                if should_add_frame:
                    # Calculate action based on actual transition proprio[t+1] - proprio[t]
                    action_t = proprio_t_plus_1 - proprio_t
                    # Override gripper action part to be absolute position from t+1
                    if joint_dim >= 2: # Avoid index error if joint_dim < 2
                         action_t[-2:] = proprio_t_plus_1[-2:]

                    # Get the images for this step 't'
                    try:
                        images_t = {
                            camera_key_mapping[raw_cam_key]: resize_with_pad(
                                np.array(Image.open(image_paths_dict[raw_cam_key][step])),
                                resize_size,
                                resize_size
                            ) for raw_cam_key in camera_keys
                        }
                    except FileNotFoundError as e:
                         print(f"Error: Image file not found during loading for step {step} in {task}: {e}. Skipping trajectory.")
                         num_frames_added_episode = 0 # Invalidate episode
                         break # Stop processing this episode
                    except Exception as e:
                         print(f"Error processing image for step {step} in {task}: {e}. Skipping trajectory.")
                         num_frames_added_episode = 0 # Invalidate episode
                         break # Stop processing this episode


                    # Add the frame to the dataset
                    dataset.add_frame(
                        {
                            "joint_position": proprio_t.astype(np.float32),
                            "actions": action_t.astype(np.float32),
                            **images_t
                        }
                    )

                    # Update the last added state
                    last_added_proprio = proprio_t
                    num_frames_added_episode += 1
                    total_frames_added += 1

            # Save the episode only if at least one frame was added
            if num_frames_added_episode > 0:
                dataset.save_episode(task=language_instruction)
                total_episodes_saved += 1
            # else:
                # print(f"Skipping saving episode {task} as no frames met the delta threshold.")


    print("\n" + "="*30)
    print("Dataset Processing Summary:")
    print(f"Total frames processed: {total_frames_processed}")
    print(f"Total frames added (after filtering): {total_frames_added}")
    print(f"Total episodes saved: {total_episodes_saved}")
    print(f"Filtering threshold (delta_threshold): {delta_threshold}")
    print("="*30 + "\n")

    # Consolidate the dataset
    print("Consolidating dataset...")
    dataset.consolidate(run_compute_stats=False) # Skip stats computation here
    print("Dataset consolidation complete.")
    print(f"Filtered dataset saved to: {output_path}")


if __name__ == "__main__":
    # Use tyro for command-line argument parsing
    tyro.cli(main)
