"""
dpgs_sim_coffee_maker_5k
- data  
    - chunk-000
        - episode_000000.parquet
        ...
    - chunk-001
        - episode_000001.parquet
        ...
    ...
- meta  
    - episodes.jsonl
        {"episode_index": 0000, "tasks": ["put the white cup on the coffee machine"], "length": 160}
        ...
    - info.json
        - {
            "codebase_version": "v2.0",
            "robot_type": "panda",
            "total_episodes": 5011,
            "total_frames": 801760,
            "total_tasks": 1,
            "total_videos": 10022,
            "total_chunks": 6,
            "chunks_size": 1000,
            "fps": 15,
            "splits": {
                "train": "0:5011"
            },
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "exterior_image_1_left": {
                    "dtype": "video",
                    "shape": [
                        224,
                        224,
                        3
                    ],
                    "names": [
                        "height",
                        "width",
                        "channel"
                    ],
                    "info": {
                        "video.fps": 15.0,
                        "video.height": 224,
                        "video.width": 224,
                        "video.channels": 3,
                        "video.codec": "av1",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": false,
                        "has_audio": false
                    }
                },
                "exterior_image_2_left": {
                    "dtype": "video",
                    "shape": [
                        224,
                        224,
                        3
                    ],
                    "names": [
                        "height",
                        "width",
                        "channel"
                    ],
                    "info": {
                        "video.fps": 15.0,
                        "video.height": 224,
                        "video.width": 224,
                        "video.channels": 3,
                        "video.codec": "av1",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": false,
                        "has_audio": false
                    }
                },
                "joint_position": {
                    "dtype": "float32",
                    "shape": [
                        16
                    ],
                    "names": [
                        "joint_position"
                    ]
                },
                "actions": {
                    "dtype": "float32",
                    "shape": [
                        16
                    ],
                    "names": [
                        "actions"
                    ]
                },
                "timestamp": {
                    "dtype": "float32",
                    "shape": [
                        1
                    ],
                    "names": null
                },
                "frame_index": {
                    "dtype": "int64",
                    "shape": [
                        1
                    ],
                    "names": null
                },
                "episode_index": {
                    "dtype": "int64",
                    "shape": [
                        1
                    ],
                    "names": null
                },
                "index": {
                    "dtype": "int64",
                    "shape": [
                        1
                    ],
                    "names": null
                },
                "task_index": {
                    "dtype": "int64",
                    "shape": [
                        1
                    ],
                    "names": null
                }
            }
        }
    - tasks.jsonl
        {"task_index": 0, "task": "put the white cup on the coffee machine"}
        ...
- videos
    - chunk-000
        - exterior_image_1_left  
            - episode_000000.mp4
            ...
        - exterior_image_2_left
            - episode_000000.mp4
            ...
    ...

# TODO: generate subsamples of the data with desired size (i.e. 1000), all with symlinks to the original data
copy over the tasks.jsonl, modify episodes.jsonl and info.json to reflect the new size

python scripts/generate_subsample_data.py --data_dir dpgs_sim_coffee_maker_5k --subsample_sizes 50 100 200 1000
# this will generate 4 new dirs:
dpgs_sim_coffee_maker_5k_subsample_50
dpgs_sim_coffee_maker_5k_subsample_100
dpgs_sim_coffee_maker_5k_subsample_200
dpgs_sim_coffee_maker_5k_subsample_1000
in the parent directory of data_dir (the directory that contains dpgs_sim_coffee_maker_5k)

Use tyro to parse the command line arguments
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
import tyro
import math

def create_symlink(src: Path, dst: Path):
    """Creates a symbolic link from src to dst, creating parent directories for dst if they don't exist.

    Args:
        src (Path): The source path (target of the link). Should be absolute.
        dst (Path): The destination path (where the link is created).
    """
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating parent directory for {dst}: {e}")
        return # Cannot proceed if parent dir creation fails

    if dst.exists() or dst.is_symlink():
        # print(f"Warning: Destination {dst} already exists. Skipping.")
        # Decide if overwriting is needed. For idempotent script runs, skipping is safer.
        # If overwriting: dst.unlink(missing_ok=True)
        pass # Skipping existing links/files
    else:
        try:
            # Use absolute path for the source to avoid broken links if the script is run from different locations
            abs_src = src.resolve()
            os.symlink(abs_src, dst, target_is_directory=abs_src.is_dir())
        except OSError as e:
            print(f"Error creating symlink from {src} ({abs_src}) to {dst}: {e}")
        except FileNotFoundError:
             print(f"Error: Source path for symlink does not exist: {src}")


def generate_subsample(data_dir: Path, subsample_size: int):
    """
    Generates a subsampled dataset directory with symbolic links to original data.

    Args:
        data_dir (Path): Path to the original dataset directory (e.g., 'dpgs_sim_coffee_maker_5k').
                         Must contain 'meta/info.json', 'meta/episodes.jsonl', 'meta/tasks.jsonl',
                         and 'data'/'videos' directories following the structure defined in info.json.
        subsample_size (int): The number of episodes to include in the subsample (episodes 0 to N-1).
    """
    if not data_dir.is_dir():
        print(f"Error: Data directory {data_dir} not found.")
        return

    parent_dir = data_dir.parent
    subsample_dir_name = f"{data_dir.name}_subsample_{subsample_size}"
    subsample_dir = parent_dir / subsample_dir_name
    print(f"Creating subsample directory: {subsample_dir}")

    # Create necessary subdirectories
    # subsample_data_dir = subsample_dir / "data"
    subsample_data_dir = subsample_dir 
    # subsample_videos_dir = subsample_dir / "videos"
    subsample_videos_dir = subsample_dir 
    subsample_meta_dir = subsample_dir / "meta"
    # Create base and meta dir; data/videos dirs created by create_symlink as needed
    subsample_meta_dir.mkdir(parents=True, exist_ok=True)

    # --- Metadata Handling ---
    original_meta_dir = data_dir / "meta"
    original_info_path = original_meta_dir / "info.json"
    original_episodes_path = original_meta_dir / "episodes.jsonl"
    original_tasks_path = original_meta_dir / "tasks.jsonl"

    if not original_info_path.is_file():
        print(f"Error: Original info.json not found at {original_info_path}")
        return
    if not original_episodes_path.is_file():
        print(f"Error: Original episodes.jsonl not found at {original_episodes_path}")
        return
    if not original_tasks_path.is_file():
         print(f"Error: Original tasks.jsonl not found at {original_tasks_path}")
         return

    # Load original info.json
    try:
        with open(original_info_path, 'r') as f:
            info_data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing original info.json: {e}")
        return

    # Process episodes.jsonl and create new one
    new_episodes_path = subsample_meta_dir / "episodes.jsonl"
    total_frames_subsample = 0
    episodes_written = 0
    original_episode_indices = [] # Store original indices we actually use

    print(f"Processing {original_episodes_path} for first {subsample_size} episodes...")
    try:
        with open(original_episodes_path, 'r') as f_in, open(new_episodes_path, 'w') as f_out:
            for line_num, line in enumerate(f_in):
                if episodes_written >= subsample_size:
                    break
                try:
                    episode_meta = json.loads(line)
                    # We assume episodes.jsonl contains metadata for episodes 0, 1, 2,... sequentially
                    current_episode_index = episode_meta.get("episode_index")
                    if current_episode_index is None:
                         print(f"Warning: Skipping line {line_num+1} in {original_episodes_path} due to missing 'episode_index'.")
                         continue

                    # Check if this episode is within the first 'subsample_size' indices
                    if current_episode_index < subsample_size:
                        # Additional check: ensure we are adding them in order if needed,
                        # or just add any episode with index < subsample_size
                        # Assuming we want exactly episodes 0 to N-1:
                        if current_episode_index == episodes_written:
                             json.dump(episode_meta, f_out)
                             f_out.write('\n')
                             total_frames_subsample += episode_meta.get("length", 0)
                             original_episode_indices.append(current_episode_index)
                             episodes_written += 1
                        else:
                             # This case handles non-sequential or out-of-order indices < subsample_size
                             # Depending on desired behavior, could error, warn, or skip.
                             print(f"Warning: Found episode index {current_episode_index} out of sequence (expected {episodes_written}) at line {line_num+1}. Including it.")
                             # If strict 0..N-1 is needed, this branch should be an error or skip.
                             # For now, we include it if < subsample_size and haven't reached count yet.
                             json.dump(episode_meta, f_out)
                             f_out.write('\n')
                             total_frames_subsample += episode_meta.get("length", 0)
                             original_episode_indices.append(current_episode_index)
                             episodes_written += 1 # Increment based on adding, not expected index

                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line {line_num+1} in {original_episodes_path}: {line.strip()}")
                except KeyError as e:
                     print(f"Warning: Skipping line {line_num+1} in {original_episodes_path} due to missing key {e}: {line.strip()}")

        # If the loop finished because EOF was reached before finding enough episodes
        if episodes_written < subsample_size:
             print(f"Warning: Only found and processed {episodes_written} episodes from {original_episodes_path}, but requested {subsample_size}.")
             # Update the target size to what was actually found
             actual_subsample_size = episodes_written
        else:
             actual_subsample_size = subsample_size

    except IOError as e:
        print(f"Error processing {original_episodes_path} or writing {new_episodes_path}: {e}")
        # Consider cleanup? shutil.rmtree(subsample_dir, ignore_errors=True)
        return

    if actual_subsample_size == 0:
         print("Error: No episodes processed. Aborting.")
         shutil.rmtree(subsample_dir, ignore_errors=True) # Clean up empty dir
         return

    # Update info.json
    info_data["total_episodes"] = actual_subsample_size
    info_data["total_frames"] = total_frames_subsample
    # Update splits - assuming a single 'train' split starting from 0
    if "splits" in info_data and "train" in info_data["splits"]:
         original_split = info_data["splits"]["train"]
         try:
             # Assuming format "start:end" or just "end" implies "0:end"
             parts = original_split.split(':')
             if len(parts) == 2:
                 start = parts[0]
                 info_data["splits"]["train"] = f"{start}:{actual_subsample_size}"
             elif len(parts) == 1: # Assumed "0:end"
                 info_data["splits"]["train"] = f"0:{actual_subsample_size}"
             else:
                 raise ValueError("Invalid split format")
         except ValueError:
              print(f"Warning: Could not parse original train split '{original_split}'. Setting to '0:{actual_subsample_size}'.")
              info_data["splits"]["train"] = f"0:{actual_subsample_size}" # Default guess

    # Update total_chunks
    chunks_size = info_data.get("chunks_size")
    if chunks_size is None or not isinstance(chunks_size, int) or chunks_size <= 0:
         print(f"Warning: Invalid or missing 'chunks_size' in info.json. Assuming 1000.")
         chunks_size = 1000
    info_data["total_chunks"] = math.ceil(actual_subsample_size / chunks_size)

    # Write updated info.json
    new_info_path = subsample_meta_dir / "info.json"
    try:
        with open(new_info_path, 'w') as f:
            json.dump(info_data, f, indent=4)
    except IOError as e:
        print(f"Error writing updated info.json to {new_info_path}: {e}")
        # Consider cleanup? shutil.rmtree(subsample_dir, ignore_errors=True)
        return

    # Copy tasks.jsonl
    new_tasks_path = subsample_meta_dir / "tasks.jsonl"
    try:
        shutil.copy2(original_tasks_path, new_tasks_path) # copy2 preserves metadata
    except IOError as e:
        print(f"Error copying {original_tasks_path} to {new_tasks_path}: {e}")
        # Consider cleanup? shutil.rmtree(subsample_dir, ignore_errors=True)
        return

    # --- Symlinking Data and Videos ---
    print(f"Creating symlinks for {actual_subsample_size} episodes...")
    data_path_template = info_data.get("data_path")
    video_path_template = info_data.get("video_path")
    video_keys = [k for k, v in info_data.get("features", {}).items() if v.get("dtype") == "video"]

    if not data_path_template:
        print("Error: 'data_path' template not found in info.json. Cannot create data symlinks.")
        # Consider cleanup? shutil.rmtree(subsample_dir, ignore_errors=True)
        return # Critical error

    if not video_path_template:
         print("Warning: 'video_path' template not found in info.json. Skipping video symlinks.")
         video_keys = [] # Don't try to link videos


    for episode_idx in original_episode_indices: # Use the list of indices we actually wrote
        episode_chunk = episode_idx // chunks_size

        # Symlink data (.parquet)
        try:
            # Format the relative path string from the template
            src_data_rel_path_str = data_path_template.format(episode_chunk=episode_chunk, episode_index=episode_idx)
            # Construct the full source path relative to the original data_dir
            src_data_path = data_dir / src_data_rel_path_str
            # Construct the full destination path relative to the subsample dir root
            dst_data_path = subsample_data_dir / src_data_rel_path_str

            create_symlink(src_data_path, dst_data_path)

        except KeyError as e:
             print(f"Error formatting data path template '{data_path_template}' with chunk={episode_chunk}, index={episode_idx}. Missing key: {e}. Skipping data link.")
        except Exception as e:
             print(f"Error processing data symlink for episode {episode_idx}: {e}")


        # Symlink videos (.mp4)
        if video_path_template and video_keys: # Check again
             for video_key in video_keys:
                  try:
                      # Format the relative path string
                      src_video_rel_path_str = video_path_template.format(episode_chunk=episode_chunk, video_key=video_key, episode_index=episode_idx)
                      # Construct full source and destination paths
                      src_video_path = data_dir / src_video_rel_path_str
                      dst_video_path = subsample_videos_dir / src_video_rel_path_str

                      create_symlink(src_video_path, dst_video_path)

                  except KeyError as e:
                       print(f"Error formatting video path template '{video_path_template}' with chunk={episode_chunk}, key={video_key}, index={episode_idx}. Missing key: {e}. Skipping video link.")
                  except Exception as e:
                       print(f"Error processing video symlink for episode {episode_idx}, key {video_key}: {e}")


    print(f"Successfully generated subsample dataset: {subsample_dir}")


def main(
    data_dir: str = "/shared/projects/icrl/data/dpgs/lerobot/mlfu7/dpgs_sim_coffee_maker_5k",
    subsample_sizes: List[int] = [50, 100, 200, 1000],
):
    """
    Generates subsampled datasets from a larger dataset directory.

    Creates new directories named <data_dir_name>_subsample_<size> in the parent
    directory of data_dir. These directories contain metadata reflecting the
    subsample size (using the first N episodes) and symbolic links pointing to the
    actual data and video files in the original dataset.

    Example usage:
        python scripts/generate_subsample_data.py --data-dir path/to/dpgs_sim_coffee_maker_5k --subsample-sizes 50 100 1000

    Args:
        data_dir: str to the original dataset directory (e.g., 'dpgs_sim_coffee_maker_5k').
                  Assumes a structure with 'data', 'videos', and 'meta' subdirectories,
                  and specific files like 'meta/info.json', 'meta/episodes.jsonl'.
        subsample_sizes: A list of desired episode counts for the subsamples (e.g., [50, 100, 1000]).
                         Each size N will create a dataset containing episodes 0 through N-1.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        # Tyro usually handles MISSING, but extra check doesn't hurt.
        print(f"Error: Data directory not found or not specified: {data_dir}")
        raise ValueError(f"Data directory not found or not specified: {data_dir}")

    if not subsample_sizes:
        print("Error: No subsample sizes provided. Use --subsample-sizes <list of ints>.")
        raise ValueError("No subsample sizes provided.")

    # Sort sizes for potentially cleaner logs, though functionality is independent
    subsample_sizes.sort()

    print(f"Generating subsamples for dataset: {data_dir}")
    print(f"Requested sizes: {subsample_sizes}")

    any_errors = False
    for size in subsample_sizes:
        if not isinstance(size, int) or size <= 0:
            print(f"Warning: Skipping invalid subsample size: {size}. Must be a positive integer.")
            any_errors = True
            continue
        print("-" * 20)
        try:
            generate_subsample(data_dir, size)
        except Exception as e:
            print(f"Error generating subsample for size {size}: {e}")
            any_errors = True
        print("-" * 20)

    if any_errors:
        print("Subsample generation finished with warnings or errors.")
    else:
        print("Subsample generation complete.")

if __name__ == "__main__":
    # Use tyro to parse command line arguments into the main function
    tyro.cli(main)