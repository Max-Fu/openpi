#!/usr/bin/env python3
"""
Generate episodes_stats.jsonl file with statistics for each episode.
This script processes parquet files and calculates min, max, mean, std, and count statistics
for each field in each episode.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import glob

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj

def calculate_array_stats(data, field_name=""):
    """Calculate statistics for array data"""
    if len(data) == 0:
        return None
    
    # Convert to numpy array for easier manipulation
    if isinstance(data[0], (list, np.ndarray)):
        # Handle multi-dimensional arrays
        arrays = np.array([np.array(item) for item in data])
        
        # Calculate statistics along the first axis (across samples)
        min_val = np.min(arrays, axis=0)
        max_val = np.max(arrays, axis=0)
        mean_val = np.mean(arrays, axis=0)
        std_val = np.std(arrays, axis=0)
    else:
        # Handle 1D arrays or scalars
        arrays = np.array(data)
        min_val = np.array([np.min(arrays)])
        max_val = np.array([np.max(arrays)])
        mean_val = np.array([np.mean(arrays)])
        std_val = np.array([np.std(arrays)])
    
    count_val = [len(data)]
    
    return {
        "min": convert_numpy_types(min_val),
        "max": convert_numpy_types(max_val),
        "mean": convert_numpy_types(mean_val),
        "std": convert_numpy_types(std_val),
        "count": count_val
    }

def calculate_image_stats(episode_index, image_type, frame_count):
    """Calculate placeholder image statistics since we don't have actual image data in parquet"""
    # Based on the example format, return placeholder values for images
    # These would normally be calculated from actual image data
    return {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[0.5]], [[0.5]], [[0.5]]],  # Placeholder values
        "std": [[[0.15]], [[0.15]], [[0.15]]],  # Placeholder values
        "count": [frame_count]
    }

def process_episode(parquet_file, episode_index):
    """Process a single episode parquet file and calculate statistics"""
    
    # Read the parquet file
    df = pd.read_parquet(parquet_file)
    
    stats = {}
    
    # Process each column in the dataframe
    for column in df.columns:
        if column in ['observation.state', 'action']:
            # These are array columns
            data = [row for row in df[column].values]
            stats[column] = calculate_array_stats(data, column)
        elif column in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']:
            # These are scalar columns
            data = df[column].values
            stats[column] = calculate_array_stats(data, column)
    
    # Add placeholder image statistics (since images aren't in parquet files)
    # Use actual frame count from the episode
    frame_count = len(df)
    stats['observation.images.front'] = calculate_image_stats(episode_index, 'front', frame_count)
    stats['observation.images.wrist'] = calculate_image_stats(episode_index, 'wrist', frame_count)
    
    return {
        "episode_index": episode_index,
        "stats": stats
    }

def main(dataset_dir: str):
    """Main function to process all episodes and generate the stats file"""
    
    # Find all parquet files
    
    data_dirs = glob.glob(f"{dataset_dir}/data/chunk-*")
    data_dirs.sort()
    for data_dir in data_dirs:
        parquet_files = glob.glob(f"{data_dir}/episode_*.parquet")
        parquet_files.sort()
    
    print(f"Found {len(parquet_files)} parquet files in {dataset_dir}")

    # Process each episode
    output_file = f"{dataset_dir}/meta/episodes_stats.jsonl"

    with open(output_file, 'w') as f:
        for parquet_file in tqdm(parquet_files, desc="Processing episodes"):
            # Extract episode index from filename
            filename = os.path.basename(parquet_file)
            episode_index = int(filename.replace('episode_', '').replace('.parquet', ''))
            
            # Process the episode
            episode_stats = process_episode(parquet_file, episode_index)
            
            # Convert to ensure JSON serializable
            episode_stats = convert_numpy_types(episode_stats)
            
            # Write to JSONL file
            f.write(json.dumps(episode_stats) + '\n')

    print(f"Generated {output_file} with statistics for {len(parquet_files)} episodes")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/mnt/amlfs-02/shared/datasets/jimmywu/2025_09_29-a4-us_v1_lerobot")
    args = parser.parse_args()
    print(f"Data dir: {args.data_dir}")
    output_file = f"{args.data_dir}/meta/episodes_stats.jsonl"
    print(f"Output file: {output_file}")

    # check if the output file exists
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Removed output file {output_file}")
        # raise ValueError(f"Output file {output_file} already exists")

    main(args.data_dir)
