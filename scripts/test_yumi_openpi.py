from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np
from openpi.policies import yumi_policy

config = config.get_config("pi0_fast_yumi")
checkpoint_dir = download.maybe_download("/home/mfu/research/openpi/checkpoints/pi0_fast_yumi/pi0_fast_yumi_finetune/29999")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
# example = {
#     "observation/exterior_image_1_left": "exterior_image_1_left",
#     "observation/exterior_image_2_left": "exterior_image_2_left",
#     "observation/joint_position": "joint_position",
#     "actions": "actions",
#     "prompt": "prompt",
# }
example = yumi_policy.make_yumi_example()
print("\n=== Example Contents ===")
print("-" * 50)
for key, value in example.items():
    print(f"\n📌 Key: {key}")
    print(f"   Type: {type(value).__name__}")
    if isinstance(value, np.ndarray):
        print(f"   Dtype: {value.dtype}")
        print(f"   Shape: {value.shape}")
    else:
        print(f"   Value: {value}")
print("-" * 50 + "\n")


action_chunk = policy.infer(example)["actions"]
print(f"Action chunk shape: {action_chunk.shape}")
print(f"Action chunk dimensions: {action_chunk.ndim}")