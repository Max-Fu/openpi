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
os.environ["LEROBOT_HOME"] = "/mnt/8tb-drive/lerobot_conversion/lerobot"
import shutil
import h5py 
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm, trange
from openpi_client.image_tools import resize_with_pad

RAW_DATASET_FOLDERS = [
    "/media/mfu/370862db-952c-481b-bba4-f681f9d8154a/mfu/icrl_new_data_0531",
    "/media/mfu/370862db-952c-481b-bba4-f681f9d8154a/mfu/icrl_new_data_0601", 
    "/media/mfu/370862db-952c-481b-bba4-f681f9d8154a/mfu/icrl_new_data_0602",
    "/media/mfu/370862db-952c-481b-bba4-f681f9d8154a/mfu/2024-06-03-new-data",
    "/media/mfu/370862db-952c-481b-bba4-f681f9d8154a/mfu/2024-06-04-new_data",
    "/home/mfu/Documents/icrl/icrl_physical/data/success/hbrl-0921",
    "/home/mfu/Documents/icrl/icrl_physical/data/success/hbrl-0926", 
    "/home/mfu/Documents/icrl/icrl_physical/data/success/hbrl_pouring_2024-11-13",
]
REPO_NAME = "mlfu7/otter_pi0_conversion"  # Name of the output dataset, also used for the Hugging Face Hub

WIRST_CAMERA_KEY = "observation/camera/image/hand_camera_left_image" # (T, 180, 320, 3) uint8
IMAGE_KEY = "observation/camera/image/varied_camera_1_left_image" # (T, 180, 320, 3) uint8
STATE_KEY = "observation/robot_state/joint_positions"
GRIPPER_STATE_KEY = "observation/robot_state/gripper_position" # (T,) (0 open 1 close)
ACTION_KEY = "action/joint_velocity"
GRIPPER_ACTION_KEY = "action/gripper_position" # (T, 1) (0 open 1 close)
VERBS = ["put", "move", "pick", "place", "stack", "grasp", "open", "close", "poke"] 
RESIZE_SIZE = 224

def main():
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    print("Dataset saved to ", output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=15,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (RESIZE_SIZE, RESIZE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (RESIZE_SIZE, RESIZE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            }, 
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in RAW_DATASET_FOLDERS:
        # get all the tasks that are collected that day 
        data_day_dir = raw_dataset_name
        print("Processing folder: ", data_day_dir)
        day_tasks = os.listdir(data_day_dir)
        for idx, task in enumerate(day_tasks):
            print(f"Task {idx}/{len(day_tasks)}: {task} is being processed")
            language_instruction = " ".join([part for part in task.split("-") if not part.isdigit()])
            if not any(verb in language_instruction.lower() for verb in VERBS):
                print(f"task {task} does not contain any verb, defaulting the verb to 'put'")
                language_instruction = "put " + language_instruction
            task_folder = f"{data_day_dir}/{task}"
            for episode in os.listdir(task_folder):
                episode_path = f"{task_folder}/{episode}/trajectory_im320_180.h5"
                with h5py.File(episode_path, "r") as f:
                    # get the length of the episode
                    episode_length = len(f[IMAGE_KEY])
                    if episode_length < 10:
                        print(f"episode {episode} is too short, skipping")
                        continue
                    for t in range(episode_length):
                        actions = np.concatenate([f[ACTION_KEY][t], np.clip(f[GRIPPER_ACTION_KEY][t], 0, 1)])
                        dataset.add_frame(
                            {
                                "exterior_image_1_left": resize_with_pad(f[IMAGE_KEY][t], RESIZE_SIZE, RESIZE_SIZE),
                                "wrist_image_left": resize_with_pad(f[WIRST_CAMERA_KEY][t], RESIZE_SIZE, RESIZE_SIZE),
                                "joint_position": f[STATE_KEY][t],
                                "gripper_position": f[GRIPPER_STATE_KEY][t],
                                "actions": actions,
                            }
                        )
                    dataset.save_episode(task=language_instruction)

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    print("Dataset saved to ", output_path)

    # Optionally push to the Hugging Face Hub
    dataset.push_to_hub(
        tags=["otter", "franka", "pi_0", "multitask"],
        private=True,
        push_videos=True,
        license="apache-2.0",
    )


if __name__ == "__main__":
    main()
