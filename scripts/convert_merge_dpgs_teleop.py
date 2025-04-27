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
from glob import glob

# --- Configuration ---
REPO_NAME = "mlfu7/dpgs_merge_sim_real_coffee_maker_1k_150"  # Name of the output dataset
RESIZE_SIZE = 224
UPTO_N_TRAJ_SIM = 10  # Limit number of sim trajectories, None for all
UPTO_N_TRAJ_TELEOP = 10 # Limit number of teleop trajectories, None for all

# Teleoperation Data Configuration
TELEOP_RAW_DATASET_FOLDERS = [
    "/shared/projects/dpgs_dataset/yumi_coffee_maker/real_data/yumi_mug_to_coffee_maker_041525_2142"
    # Add more teleop folders if needed
]
TELEOP_LANGUAGE_INSTRUCTIONS = [
    "put the white cup on the coffee machine"
    # Match instructions to folders
]
TELEOP_CAMERA_KEYS = ["_left", "_right"]
TELEOP_CAMERA_KEY_MAPPING = {
    "_left": "exterior_image_1_left",
    "_right": "exterior_image_2_left",
}
TELEOP_STATE_KEY = "_joint.zarr" # Relative to trajectory base name

# Simulation Data Configuration
SIM_RAW_DATASET_FOLDERS = [
    "/shared/projects/dpgs_dataset/yumi_coffee_maker/successes_041525_1620",
    # Add more sim folders if needed
]
SIM_LANGUAGE_INSTRUCTIONS = [
    "put the white cup on the coffee machine"
    # Match instructions to folders
]
SIM_CAMERA_KEYS = ["camera_0/rgb", "camera_1/rgb"]
SIM_CAMERA_KEY_MAPPING = {
    "camera_0/rgb": "exterior_image_1_left",
    "camera_1/rgb": "exterior_image_2_left",
}
SIM_STATE_KEY = "robot_data/robot_data_joint.zarr" # Relative to task folder

# --- Helper Function for Joint Conversion ---

def convert_robot_to_model_format(robot_joints: np.ndarray) -> np.ndarray:
    """
    Update the current proprioception state by mapping the robot's joint order to the model's format.
    
    Args:
        robot_joints (np.ndarray): Joint values in robot format (shape: 16,).
            Robot format:
            yumi_joint_1_r, yumi_joint_2_r, yumi_joint_7_r, yumi_joint_3_r,
            yumi_joint_4_r, yumi_joint_5_r, yumi_joint_6_r, yumi_joint_1_l,
            yumi_joint_2_l, yumi_joint_7_l, yumi_joint_3_l, yumi_joint_4_l,
            yumi_joint_5_l, yumi_joint_6_l, gripper_r_joint, gripper_l_joint
    
    Returns:
        np.ndarray: Joint values in model format (shape: 16,).
            Model format:
            yumi_joint_1_l, yumi_joint_1_r, yumi_joint_2_l, yumi_joint_2_r, 
            yumi_joint_7_l, yumi_joint_7_r, yumi_joint_3_l, yumi_joint_3_r,
            yumi_joint_4_l, yumi_joint_4_r, yumi_joint_5_l, yumi_joint_5_r,
            yumi_joint_6_l, yumi_joint_6_r, gripper_l_joint, gripper_r_joint
    """
    model_joints = np.zeros(16)
    
    # Map left arm joints
    model_joints[0] = robot_joints[7]  # yumi_joint_1_l
    model_joints[2] = robot_joints[8]  # yumi_joint_2_l
    model_joints[4] = robot_joints[9]  # yumi_joint_7_l
    model_joints[6] = robot_joints[10] # yumi_joint_3_l
    model_joints[8] = robot_joints[11] # yumi_joint_4_l
    model_joints[10] = robot_joints[12] # yumi_joint_5_l
    model_joints[12] = robot_joints[13] # yumi_joint_6_l
    
    # Map right arm joints
    model_joints[1] = robot_joints[0]  # yumi_joint_1_r
    model_joints[3] = robot_joints[1]  # yumi_joint_2_r
    model_joints[5] = robot_joints[2]  # yumi_joint_7_r
    model_joints[7] = robot_joints[3]  # yumi_joint_3_r
    model_joints[9] = robot_joints[4]  # yumi_joint_4_r
    model_joints[11] = robot_joints[5] # yumi_joint_5_r
    model_joints[13] = robot_joints[6] # yumi_joint_6_r
    
    # Map gripper joints
    model_joints[14] = robot_joints[15] # gripper_l_joint
    model_joints[15] = robot_joints[14] # gripper_r_joint
    
    assert model_joints.shape == (16,)
    return model_joints

# --- Main Conversion Logic ---

def main():
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    print("Dataset saved to ", output_path)

    # Create LeRobot dataset, define features to store
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="yumi", # Assuming Yumi robot
        fps=15, # Check actual FPS if different
        features={
            "exterior_image_1_left": {
                "dtype": "video",
                "shape": (RESIZE_SIZE, RESIZE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "video",
                "shape": (RESIZE_SIZE, RESIZE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (16,),
                "names": ["joint_position"], # Should be in model format
            },
            "actions": {
                "dtype": "float32",
                "shape": (16,),
                "names": ["actions"], # Delta actions in model format
            },
        },
        image_writer_threads=20,
        image_writer_processes=10,
    )

    total_episodes = 0

    # --- Process Simulation Data ---
    print("\nProcessing Simulation Data...")
    for raw_dataset_folder, language_instruction in zip(SIM_RAW_DATASET_FOLDERS, SIM_LANGUAGE_INSTRUCTIONS):
        print(f"Processing sim folder: {raw_dataset_folder}")
        try:
            trajs = sorted(os.listdir(raw_dataset_folder))
        except FileNotFoundError:
            print(f"Warning: Sim folder not found: {raw_dataset_folder}. Skipping.")
            continue
            
        if UPTO_N_TRAJ_SIM is not None:
            trajs = trajs[:UPTO_N_TRAJ_SIM]
            
        for idx, task in tqdm(enumerate(trajs), total=len(trajs), desc="Sim Trajectories"):
            # print(f"Sim Trajectory {idx}/{len(trajs)}: {task} is being processed")
            task_folder = os.path.join(raw_dataset_folder, task)
            state_file = os.path.join(task_folder, SIM_STATE_KEY)
            
            if not os.path.exists(state_file):
                print(f"Warning: State file not found for sim task {task}: {state_file}. Skipping.")
                continue
                
            try:
                proprio_data = zarr.load(state_file)
            except Exception as e:
                print(f"Error loading sim state file {state_file}: {e}. Skipping.")
                continue

            seq_length = proprio_data.shape[0] - 1
            if seq_length <= 0:
                print(f"Warning: Sim trajectory {task} has insufficient length ({seq_length+1}). Skipping.")
                continue

            # Check if image folders exist and get image paths
            images = {}
            valid_trajectory = True
            for key in SIM_CAMERA_KEYS:
                img_folder = os.path.join(task_folder, key)
                if not os.path.isdir(img_folder):
                    print(f"Warning: Image folder not found for sim task {task}: {img_folder}. Skipping trajectory.")
                    valid_trajectory = False
                    break
                try:
                    img_files = sorted(os.listdir(img_folder))
                    if len(img_files) < seq_length:
                         print(f"Warning: Insufficient images found for sim task {task} camera {key}. Expected {seq_length}, found {len(img_files)}. Skipping trajectory.")
                         valid_trajectory = False
                         break
                    images[key] = [os.path.join(img_folder, f) for f in img_files]
                except Exception as e:
                     print(f"Error listing images for sim task {task} camera {key}: {e}. Skipping trajectory.")
                     valid_trajectory = False
                     break
            if not valid_trajectory:
                continue

            images_per_step = [
                {key: images[key][i] for key in SIM_CAMERA_KEYS} for i in range(seq_length)
            ]

            # Add frames for the current trajectory
            for step in range(seq_length):
                proprio_t = proprio_data[step]
                proprio_t_plus_1 = proprio_data[step + 1]
                action_t = proprio_t_plus_1 - proprio_t # Simple delta
                action_t[-2:] = proprio_t_plus_1[-2:]
                # Get and resize images for this step
                images_t = {}
                valid_step = True
                for key in SIM_CAMERA_KEYS:
                    try:
                        img = Image.open(images_per_step[step][key])
                        images_t[SIM_CAMERA_KEY_MAPPING[key]] = resize_with_pad(
                            np.array(img), RESIZE_SIZE, RESIZE_SIZE
                        )
                    except Exception as e:
                        print(f"Error processing image {images_per_step[step][key]} for sim task {task}: {e}. Skipping step.")
                        valid_step = False
                        break
                if not valid_step:
                    continue # Skip frame if image loading fails
                    
                dataset.add_frame(
                    {
                        "joint_position": proprio_t, # Sim data is already in model format
                        "actions": action_t,
                        **images_t
                    }
                )
            dataset.save_episode(task=language_instruction)
            total_episodes += 1

    # --- Process Teleoperation Data ---
    print("\nProcessing Teleoperation Data...")
    for raw_dataset_folder, language_instruction in zip(TELEOP_RAW_DATASET_FOLDERS, TELEOP_LANGUAGE_INSTRUCTIONS):
        print(f"Processing teleop folder: {raw_dataset_folder}")
        try:
            # Find trajectories based on the state key pattern
            traj_base_names = sorted([i.replace(TELEOP_STATE_KEY, "") for i in glob(f"{raw_dataset_folder}/*{TELEOP_STATE_KEY}")])
        except FileNotFoundError:
             print(f"Warning: Teleop folder not found: {raw_dataset_folder}. Skipping.")
             continue
             
        if UPTO_N_TRAJ_TELEOP is not None:
            traj_base_names = traj_base_names[:UPTO_N_TRAJ_TELEOP]
            
        for idx, task_base_name in tqdm(enumerate(traj_base_names), total=len(traj_base_names), desc="Teleop Trajectories"):
            # print(f"Teleop Trajectory {idx}/{len(traj_base_names)}: {task_base_name} is being processed")
            state_file = task_base_name + TELEOP_STATE_KEY
            
            if not os.path.exists(state_file):
                 print(f"Warning: State file not found for teleop task {task_base_name}: {state_file}. Skipping.")
                 continue
                 
            try:
                proprio_data_robot_format = zarr.load(state_file)
            except Exception as e:
                print(f"Error loading teleop state file {state_file}: {e}. Skipping.")
                continue

            seq_length = proprio_data_robot_format.shape[0] - 1
            if seq_length <= 0:
                print(f"Warning: Teleop trajectory {task_base_name} has insufficient length ({seq_length+1}). Skipping.")
                continue
                
            # Check if image folders exist and get image paths
            images = {}
            valid_trajectory = True
            for key in TELEOP_CAMERA_KEYS:
                img_folder = task_base_name + key
                if not os.path.isdir(img_folder):
                    print(f"Warning: Image folder not found for teleop task {task_base_name}: {img_folder}. Skipping trajectory.")
                    valid_trajectory = False
                    break
                try:
                    img_files = sorted(os.listdir(img_folder))
                    if len(img_files) < seq_length:
                        print(f"Warning: Insufficient images found for teleop task {task_base_name} camera {key}. Expected {seq_length}, found {len(img_files)}. Skipping trajectory.")
                        valid_trajectory = False
                        break
                    images[key] = [os.path.join(img_folder, f) for f in img_files]
                except Exception as e:
                    print(f"Error listing images for teleop task {task_base_name} camera {key}: {e}. Skipping trajectory.")
                    valid_trajectory = False
                    break
            if not valid_trajectory:
                continue

            images_per_step = [
                {key: images[key][i] for key in TELEOP_CAMERA_KEYS} for i in range(seq_length)
            ]
            
            # Convert entire trajectory proprio to model format first for efficiency
            proprio_data_model_format = np.array([convert_robot_to_model_format(p) for p in proprio_data_robot_format])

            # Add frames for the current trajectory
            for step in range(seq_length):
                proprio_t_model = proprio_data_model_format[step]
                proprio_t_plus_1_model = proprio_data_model_format[step + 1]
                action_t_model = proprio_t_plus_1_model - proprio_t_model # Simple delta in model format
                action_t_model[-2:] = proprio_t_plus_1_model[-2:]
                
                # Get and resize images for this step
                images_t = {}
                valid_step = True
                for key in TELEOP_CAMERA_KEYS:
                    try:
                        img = Image.open(images_per_step[step][key])
                        images_t[TELEOP_CAMERA_KEY_MAPPING[key]] = resize_with_pad(
                            np.array(img), RESIZE_SIZE, RESIZE_SIZE
                        )
                    except Exception as e:
                        print(f"Error processing image {images_per_step[step][key]} for teleop task {task_base_name}: {e}. Skipping step.")
                        valid_step = False
                        break
                if not valid_step:
                    continue # Skip frame if image loading fails
                    
                dataset.add_frame(
                    {
                        "joint_position": proprio_t_model, # Use model format proprio
                        "actions": action_t_model,      # Use model format action
                        **images_t
                    }
                )
            dataset.save_episode(task=language_instruction)
            total_episodes += 1

    # Consolidate the dataset
    if total_episodes > 0:
        print(f"\nConsolidating dataset with {total_episodes} episodes...")
        dataset.consolidate(run_compute_stats=False) # Skip stats computation for now
        print("Dataset saved to ", output_path)
    else:
        print("\nNo episodes were added to the dataset. Skipping consolidation.")
        # Optionally remove the empty dataset directory
        if output_path.exists():
             shutil.rmtree(output_path)
             print(f"Removed empty dataset directory: {output_path}")

if __name__ == "__main__":
    main()
