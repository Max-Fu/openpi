export LEROBOT_HOME=/shared/projects/icrl/data/dpgs/lerobot # bajcsy path
CUDA_VISIBLE_DEVICES=0 uv run scripts/convert_dpgs_data.py
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name pi0_fast_yumi