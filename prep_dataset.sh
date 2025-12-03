#!/bin/bash

# Maximum number of parallel jobs (adjust based on your system resources)
MAX_JOBS=${MAX_JOBS:-8}

# Function to process a single dataset
process_dataset() {
    local dataset_path=$1
    local TASK_ID=$(basename "$dataset_path")
    local OUTPUT_PATH=assets/pi05_yam_lora/$TASK_ID/norm_stats.json
    echo "Processing: $TASK_ID"
    uv run python scripts/remove_eef_from_pkl.py --folder-path "$dataset_path"
    uv run scripts/generate_episode_stats.py --data-dir "$dataset_path"
    uv run scripts/generate_norm_stats.py --data-path "$dataset_path" --output-path "$OUTPUT_PATH"
    echo "Completed: $TASK_ID"
}

export -f process_dataset

# Collect all dataset paths
dataset_paths=()
for dataset_path in /mnt/amlfs-02/shared/datasets/yam_v5/*; do
    if [ -d "$dataset_path" ]; then
        dataset_paths+=("$dataset_path")
    fi
done

echo "Found ${#dataset_paths[@]} datasets to process"

# Process datasets in parallel with job control
for dataset_path in "${dataset_paths[@]}"; do
    # Wait if we've reached the maximum number of background jobs
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 0.1
    done
    
    # Start processing in background
    process_dataset "$dataset_path" &
done

# Wait for all remaining background jobs to complete
wait

echo "All datasets processed successfully!"