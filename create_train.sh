GLOBAL_BATCH_SIZE=512
CHECKPOINT_BASE_DIR=/mnt/amlfs-02/shared/checkpoints/maxf/openpi
TOTAL_STEPS=10000
train_script_root="yam_train_scripts"
mkdir -p $train_script_root

# Collect all dataset paths
dataset_paths=()
for dataset_path in /mnt/amlfs-02/shared/datasets/yam_v5/*; do
    if [ -d "$dataset_path" ]; then
        dataset_paths+=("$dataset_path")
    fi
done

echo "Found ${#dataset_paths[@]} datasets to process"

for dataset_path in "${dataset_paths[@]}"; do
    echo "creating train script for: $dataset_path"
    repo_id=$(basename $dataset_path)
    train_script="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    uv run scripts/train.py pi05_yam_lora \
    --data.repo-id $repo_id  \
    --data.use-delta-joint-actions \
    --data.root $dataset_path \
    --save-interval 1000 \
    --keep-period 2500 \
    --log-interval 20 \
    --num-train-steps $TOTAL_STEPS \
    --model.action-horizon 40 \
    --exp-name $repo_id \
    --checkpoint-base-dir $CHECKPOINT_BASE_DIR \
    --num-workers 40 \
    --batch-size $GLOBAL_BATCH_SIZE \
    --overwrite"
    echo $train_script > $train_script_root/train_$repo_id.sh
done

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
#     uv run scripts/train.py pi05_yam_lora \
#     --data.repo-id a5_us_v2  \
#     --data.use-delta-joint-actions \
#     --data.root /mnt/amlfs-02/shared/datasets/jimmywu/2025_11_07-a5_us_v2 \
#     --save-interval 1000 \
#     --keep-period 2500 \
#     --log-interval 20 \
#     --num-train-steps $TOTAL_STEPS \
#     --model.action-horizon 40 \
#     --exp-name=pi05_yam_a5_us_new_lora_delta_act40_10k_gb480 \
#     --checkpoint-base-dir $CHECKPOINT_BASE_DIR \
#     --num-workers 40 \
#     --batch-size $GLOBAL_BATCH_SIZE \
#     --overwrite