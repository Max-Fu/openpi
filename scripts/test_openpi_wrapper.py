import time
from tqdm import tqdm
import numpy as np
from eval_wrapper import OpenPIWrapper
import tyro
from dataclasses import dataclass
from typing import Optional

@dataclass
class TestConfig:
    model_ckpt_folder: str = "/home/mfu/research/openpi/checkpoints/pi0_fast_yumi/pi0_fast_yumi_finetune"
    ckpt_id: int = 29999
    # ckpt_id: int = 30001
    text_prompt: str = "put the white cup on the coffee machine"
    batch_size: int = 1
    sequence_length: int = 10
    num_cameras: int = 2
    image_height: int = 270
    image_width: int = 480
    num_joints: int = 14
    num_grippers: int = 2

def create_dummy_batch(config: TestConfig) -> dict:
    """Create a dummy batch with correct dimensions and dtypes."""
    # Create observation data (B, T, num_cameras, H, W, C)
    observation = np.random.randint(
        0, 255, 
        size=(config.batch_size, config.sequence_length, config.num_cameras, 
              config.image_height, config.image_width, 3), 
        dtype=np.uint8
    )
    
    # Create proprioceptive data (B, T, num_joints + num_grippers)
    proprio = np.random.randn(
        config.batch_size, 
        config.sequence_length, 
        config.num_joints + config.num_grippers
    ).astype(np.float64)
    
    return {
        "observation": observation,
        "proprio": proprio
    }

def main(config: TestConfig):
    # Initialize the wrapper
    wrapper = OpenPIWrapper(
        model_ckpt_folder=config.model_ckpt_folder,
        ckpt_id=config.ckpt_id,
        text_prompt=config.text_prompt
    )
    
    # Create dummy batch
    nbatch = create_dummy_batch(config)
    
    # Print input shapes and dtypes
    print("\nInput shapes and dtypes:")
    print(f"observation shape: {nbatch['observation'].shape}, dtype: {nbatch['observation'].dtype}")
    print(f"proprio shape: {nbatch['proprio'].shape}, dtype: {nbatch['proprio'].dtype}")
    
    try:
        # Get action from wrapper
        # Time the function for 10 runs, skipping first warm-up run
        # Warm-up run
        _ = wrapper(nbatch)
        
        # Time 10 runs
        times = []
        for _ in tqdm(range(10), desc="Timing inference"):
            nbatch = create_dummy_batch(config)
            start = time.time()
            target_joint_positions = wrapper(nbatch)
            times.append(time.time() - start)
            
        avg_time = sum(times) / len(times)
        print(f"\nAverage inference time: {avg_time:.4f} seconds")
        
        # Print output shapes and dtypes
        print("\nOutput shapes and dtypes:")
        print(f"target_joint_positions shape: {target_joint_positions.shape}, dtype: {target_joint_positions.dtype}")
        
        # Verify output dimensions
        expected_shape = (10, config.num_joints + config.num_grippers)  # (sequence_length, num_joints + num_grippers)
        assert target_joint_positions.shape == expected_shape, f"Expected shape {expected_shape}, got {target_joint_positions.shape}"
        print("\nTest passed successfully!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    args = tyro.cli(TestConfig)
    main(args)