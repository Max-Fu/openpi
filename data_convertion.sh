export LEROBOT_HOME=/mnt/disks/ssd1/lerobot
CUDA_VISIBLE_DEVICES=9 uv run scripts/convert_dpgs_data.py
CUDA_VISIBLE_DEVICES=9 uv run scripts/compute_norm_stats.py --config-name pi0_fast_yumi