# Running R2R2R Data

# Installation

```bash
git clone --recurse-submodules git@github.com:Max-Fu/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

Then install the repo

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

# (Optional if home directory is large enough) Create Data Path 

```bash
mkdir lerobot_conversion/lerobot # create a path to save lerobot data, just in case home directory is not enough
export LEROBOT_HOME=lerobot_conversion/lerobot
```

# Configs 
There are the following configs for the different datasets:

Drawer task
```bash 
pi0_fast_sim_yumi_drawer_50
pi0_fast_sim_yumi_drawer_100
pi0_fast_sim_yumi_drawer_150
pi0_fast_sim_yumi_drawer_1k
```

Faucet task
```bash 
pi0_fast_sim_yumi_faucet_50
pi0_fast_sim_yumi_faucet_100
pi0_fast_sim_yumi_faucet_150
pi0_fast_sim_yumi_faucet_1k
```

LED Light task
```bash
pi0_fast_sim_yumi_led_50
pi0_fast_sim_yumi_led_100
pi0_fast_sim_yumi_led_150
pi0_fast_sim_yumi_led_1k
```

For more details, please refer to [src/openpi/training/config.py](src/openpi/training/config.py).

# To launch training for each of these tasks: 
```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name pi0_fast_sim_yumi_drawer_50
uv run scripts/train.py pi0_fast_sim_yumi_drawer_50 --exp-name=pi0_fast_sim_yumi_drawer_50 --checkpoint-base-dir /path/to/checkpoint-base-dir --overwrite --fsdp_devices 1
```

for Multi-GPU training, update `fsdp_devices` to the number of GPUs you have. 
```bash
uv run scripts/train.py pi0_fast_sim_yumi_drawer_50 --exp-name=pi0_fast_sim_yumi_drawer_50 --checkpoint-base-dir /path/to/checkpoint-base-dir --overwrite --fsdp_devices 2
```

for resuming training, change the `--overwrite` argument to `--resume`. 
```bash
uv run scripts/train.py pi0_fast_sim_yumi_drawer_50 --exp-name=pi0_fast_sim_yumi_drawer_50 --checkpoint-base-dir /path/to/checkpoint-base-dir --resume --fsdp_devices 1
```


