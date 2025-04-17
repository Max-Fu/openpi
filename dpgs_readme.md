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

# Creating Data Path 

```bash
# create a path to save lerobot data, usually home directory are small
mkdir lerobot_conversion/lerobot 
export LEROBOT_HOME=lerobot_conversion/lerobot
cd $LEROBOT_HOME

# Download the dataset
huggingface-cli download mlfu7/dpgs_real_coffee_maker_150 --repo-type dataset --local-dir dpgs_real_coffee_maker_150
huggingface-cli download mlfu7/dpgs_sim_faucet_5k --repo-type dataset --local-dir dpgs_sim_faucet_5k
huggingface-cli download mlfu7/dpgs_sim_led_5k --repo-type dataset --local-dir dpgs_sim_led_5k

# ... we will add more here when we generate them
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

First you need to compute the normalization stats for the dataset. 

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name pi0_fast_sim_yumi_drawer_50
```
To avoid recomputation, copy the stats: 
```bash
cp -r assets/pi0_fast_sim_yumi_drawer_50 pi0_fast_sim_yumi_drawer_100
cp -r assets/pi0_fast_sim_yumi_drawer_50 pi0_fast_sim_yumi_drawer_150
cp -r assets/pi0_fast_sim_yumi_drawer_50 pi0_fast_sim_yumi_drawer_1k
```

Now we can launch training for each of these tasks, remember to update the `checkpoint-base-dir` to the path you want to save the checkpoints: 
```bash
export LEROBOT_HOME=/path/to/lerobot/home
uv run scripts/train.py pi0_fast_sim_yumi_drawer_50 --exp-name=pi0_fast_sim_yumi_drawer_50 --checkpoint-base-dir /path/to/checkpoint-base-dir --overwrite --fsdp_devices 1
```

for Multi-GPU training (i.e. A100 40GB where the GPU memory is not enough), update `fsdp_devices` to the number of GPUs you have. 
```bash
export LEROBOT_HOME=/path/to/lerobot/home
uv run scripts/train.py pi0_fast_sim_yumi_drawer_50 --exp-name=pi0_fast_sim_yumi_drawer_50 --checkpoint-base-dir /path/to/checkpoint-base-dir --overwrite --fsdp_devices 2
```

for resuming training, change the `--overwrite` argument to `--resume`. 
```bash
export LEROBOT_HOME=/path/to/lerobot/home
uv run scripts/train.py pi0_fast_sim_yumi_drawer_50 --exp-name=pi0_fast_sim_yumi_drawer_50 --checkpoint-base-dir /path/to/checkpoint-base-dir --resume --fsdp_devices 1
```


