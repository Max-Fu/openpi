#!/usr/bin/env python3
"""
Generate norm_stats.json from stats.json
This script extracts normalization statistics for state and actions from the global stats.
"""

import json

def main(stats_path, output_path):
    # Load the existing stats.json file
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    # Extract state statistics (observation.state)
    state_stats = {
        "mean": stats["observation.state"]["mean"],
        "std": stats["observation.state"]["std"],
        "q01": stats["observation.state"]["q01"],
        "q99": stats["observation.state"]["q99"]
    }
    
    # Extract action statistics
    action_stats = {
        "mean": stats["action"]["mean"],
        "std": stats["action"]["std"],
        "q01": stats["action"]["q01"],
        "q99": stats["action"]["q99"]
    }
    
    # Create the norm_stats structure
    norm_stats = {
        "norm_stats": {
            "state": state_stats,
            "actions": action_stats  # Note: "actions" plural to match the desired format
        }
    }
    
    # Write to norm_stats.json
    with open(output_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    
    print("Generated norm_stats.json successfully!")
    print(f"State dimensions: {len(state_stats['mean'])}")
    print(f"Action dimensions: {len(action_stats['mean'])}")


if __name__ == "__main__":
    import os
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/mnt/amlfs-02/shared/datasets/jimmywu/2025_09_29-a4-us_v1_lerobot")
    parser.add_argument("--output-path", type=str, default="assets/pi05_yam_lora/a4/norm_stats.json")
    args = parser.parse_args()
    
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    
    # check if the output path exists
    if os.path.exists(args.output_path):
        raise ValueError(f"Output path {args.output_path} already exists")
    else:
        # create the output directory
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # stats is in DATASET_PATH/meta/stats.json
    stats_path = os.path.join(args.data_path, "meta", "stats.json")
    
    # check if the stats path exists
    if not os.path.exists(stats_path):
        raise ValueError(f"Stats path {stats_path} does not exist")

    main(stats_path, args.output_path)
