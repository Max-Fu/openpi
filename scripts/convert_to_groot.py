import pandas as pd 
from glob import glob 
from tqdm import tqdm

def main(
    dataset_root_path : str = "/home/mfu/dataset/groot/dpgs_conversion_video_groot",
):
    """
    Find all parquet files and copy task_index column to annotation.human.action.task_description
    
    Args:
        dataset_root_path: str - Root path to search for parquet files
    """
    # Find all parquet files recursively
    parquet_files = glob(f"{dataset_root_path}/**/*.parquet", recursive=True)
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Process each file
    for parquet_file in tqdm(parquet_files):
        print(f"Processing {parquet_file}")
        
        # Read parquet file
        df = pd.read_parquet(parquet_file)
        
        # Copy task_index column to new name
        df['annotation.human.action.task_description'] = df['task_index']
        # df['observation.state'] = df['joint_position']
        # df['action'] = df['actions']

        # Save back to same file
        df.to_parquet(parquet_file)
        print(f"Updated {parquet_file}")
        
if __name__ == "__main__":
    main()