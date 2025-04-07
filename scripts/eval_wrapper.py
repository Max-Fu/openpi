import os 
import numpy as np
import torch
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np
from openpi.policies import yumi_policy
from PIL import Image
from openpi_client.image_tools import resize_with_pad

RESIZE_SIZE = 224

class DiffusionWrapper():
    def __init__(
        self, 
        model_ckpt_folder : str, 
        ckpt_id : int, 
        text_prompt : str = "put the white cup on the coffee machine",
    ) -> None:
        """
        Args:
            model_ckpt_folder: str, path to the model checkpoint folder
            ckpt_id: int, checkpoint id
            device: str, device to run the model on
            text_prompt: str, text prompt to use for the model
        Example:
        model_ckpt_folder = "/home/mfu/research/openpi/checkpoints/pi0_fast_yumi/pi0_fast_yumi_finetune"
        ckpt_id = 29999
        device = "cuda"
        """
        config = config.get_config("pi0_fast_yumi")
        checkpoint_dir = os.path.join(model_ckpt_folder, f"{ckpt_id}")
        # Create a trained policy.
        self.policy = policy_config.create_trained_policy(config, checkpoint_dir)
        self.text_prompt = text_prompt

    def __call__(self, nbatch):
        # TODO reformat data into the correct format for the model
        # TODO: communicate with justin that we are using numpy to pass the data. Also we are passing in uint8 for images 
        """
        Model input expected: 
            📌 Key: observation/exterior_image_1_left
            Type: ndarray
            Dtype: uint8
            Shape: (224, 224, 3)

            📌 Key: observation/exterior_image_2_left
            Type: ndarray
            Dtype: uint8
            Shape: (224, 224, 3)

            📌 Key: observation/joint_position
            Type: ndarray
            Dtype: float64
            Shape: (16,)

            📌 Key: prompt
            Type: str
            Value: do something
        
        Model will output:
            📌 Key: actions
            Type: ndarray
            Dtype: float64
            Shape: (10, 16)
        """
        # update nbatch observation (B, T, num_cameras, H, W, C) -> (B, num_cameras, H, W, C)
        nbatch["observation"] = nbatch["observation"][:, -1] # only use the last observation step
        if nbatch["observation"].shape[-1] != 3:
            # make B, num_cameras, H, W, C  from B, num_cameras, C, H, W
            # permute if pytorch
            nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))

        # nbatch["proprio"] is B, T, 16, where B=1
        joint_positions = nbatch["proprio"][0, -1]
        batch = {
            "observation/exterior_image_1_left": resize_with_pad(
                nbatch["observation"][0, 0], 
                RESIZE_SIZE,
                RESIZE_SIZE
            ),
            "observation/exterior_image_2_left": resize_with_pad(
                nbatch["observation"][0, 1], 
                RESIZE_SIZE,
                RESIZE_SIZE
            ),
            "observation/joint_position": joint_positions,
            "prompt": self.text_prompt,
        }
        action = self.policy.infer(batch)
        # remove the gripper command from the joint position, since the model is predicting delta joint positions
        # and absolute gripper positions 
        joint_positions[-2:] = 0
        # convert to absolute action and append gripper command
        # action["actions"] shape: (10, 16), joint_positions shape: (16,)
        # Need to broadcast joint_positions to match action sequence length
        target_joint_positions = action["actions"] + joint_positions[None, :]  # Shape: (10, 16)
        return target_joint_positions