#!/bin/bash
# Array of checkpoints to download
checkpoints_names=(
    "GearCheckpoints/trinity-ckpt-repository/openpi_baselines/xdof.a2_2025-11-04_19-28-44_USA/9999/*"
    "GearCheckpoints/trinity-ckpt-repository/openpi_baselines/xdof.a3_2025-11-04_19-32-10_USA/9999/*"
    "GearCheckpoints/trinity-ckpt-repository/openpi_baselines/xdof.a4_2025-11-04_19-35-33_USA/9999/*"
    "GearCheckpoints/trinity-ckpt-repository/openpi_baselines/xdof.a5_2025-11-04_19-38-51_USA/9999/*"
    "GearCheckpoints/trinity-ckpt-repository/openpi_baselines/xdof.h1_2025-11-04_19-42-08_USA/9999/*"
    "GearCheckpoints/trinity-ckpt-repository/openpi_baselines/xdof.h2_2025-11-04_19-45-27_USA/9999/*"
    "GearCheckpoints/trinity-ckpt-repository/openpi_baselines/xdof.h2beta_2025-11-25_18-14-35_v4_USA/9999/*"
    "GearCheckpoints/trinity-ckpt-repository/openpi_baselines/xdof.h5_2025-11-04_19-48-48_USA/9999/*"
    "GearCheckpoints/trinity-ckpt-repository/openpi_baselines/xdof.i1_2025-11-04_19-52-04_USA/9999/*"
)

# Maximum number of parallel downloads
MAX_PARALLEL=9

# Counter for managing parallel jobs
count=0

echo "Starting parallel download of ${#checkpoints_names[@]} checkpoints with max ${MAX_PARALLEL} parallel jobs..."

# Download all checkpoints in parallel
for ckpt in "${checkpoints_names[@]}"; do
    # Extract the folder name (last part after the last /)
    folder_name=$(echo "$ckpt" | sed 's|.*/openpi_baselines/||' | sed 's|/\*$||')

    echo "Starting download: $folder_name"
    # echo "gear data download $ckpt $folder_name"
    gear data download "$ckpt" "$folder_name" &

    ((count++))

    # If we've reached max parallel jobs, wait for them to complete
    if (( count >= MAX_PARALLEL )); then
        wait -n  # Wait for any one job to complete
        ((count--))
    fi
done

# Wait for all remaining background jobs to complete