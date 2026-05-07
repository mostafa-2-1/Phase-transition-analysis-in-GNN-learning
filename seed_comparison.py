# # # import os
# # # import shutil
# # # from pathlib import Path

# # # def copy_missing_files(source_dir, target_dir):
# # #     """
# # #     Copy files from source directory to target directory if they don't exist in target.
    
# # #     Args:
# # #         source_dir (str): Path to source directory (tsplib_data)
# # #         target_dir (str): Path to target directory (graphss_chosen)
# # #     """
    
# # #     # Convert to Path objects for easier handling
# # #     source_path = Path(source_dir)
# # #     target_path = Path(target_dir)
    
# # #     # Check if source directory exists
# # #     if not source_path.exists():
# # #         print(f"Error: Source directory '{source_dir}' does not exist.")
# # #         return
    
# # #     # Check if target directory exists, create if it doesn't
# # #     if not target_path.exists():
# # #         print(f"Target directory '{target_dir}' does not exist. Creating it...")
# # #         target_path.mkdir(parents=True, exist_ok=True)
    
# # #     # Get all files from source directory
# # #     try:
# # #         source_files = [f for f in source_path.iterdir() if f.is_file()]
# # #     except PermissionError:
# # #         print(f"Error: Permission denied to read source directory '{source_dir}'.")
# # #         return
    
# # #     if not source_files:
# # #         print(f"No files found in source directory '{source_dir}'.")
# # #         return
    
# # #     # Track statistics
# # #     files_copied = 0
# # #     files_skipped = 0
    
# # #     print(f"Checking files from '{source_dir}' against '{target_dir}'...")
# # #     print("-" * 50)
    
# # #     # Check each file in source
# # #     for source_file in source_files:
# # #         target_file = target_path / source_file.name
        
# # #         if not target_file.exists():
# # #             # File doesn't exist in target, copy it
# # #             try:
# # #                 shutil.copy2(source_file, target_file)
# # #                 print(f"✓ Copied: {source_file.name}")
# # #                 files_copied += 1
# # #             except (shutil.Error, PermissionError, OSError) as e:
# # #                 print(f"✗ Failed to copy {source_file.name}: {e}")
# # #                 files_skipped += 1
# # #         else:
# # #             print(f"  Skipped (already exists): {source_file.name}")
# # #             files_skipped += 1
    
# # #     # Print summary
# # #     print("-" * 50)
# # #     print(f"Summary:")
# # #     print(f"  Files copied: {files_copied}")
# # #     print(f"  Files skipped: {files_skipped}")
# # #     print(f"  Total files in source: {len(source_files)}")

# # # def main():
# # #     # Define folder names
# # #     folder_a = "tsplib_data"      # Source folder
# # #     folder_b = "graphss_chosen"   # Target folder
    
# # #     # Get the current script's directory
# # #     current_dir = Path.cwd()
    
# # #     # Full paths (assuming folders are in the same directory as the script)
# # #     source_dir = current_dir / folder_a
# # #     target_dir = current_dir / folder_b
    
# # #     print(f"Script started...")
# # #     print(f"Source directory: {source_dir}")
# # #     print(f"Target directory: {target_dir}")
# # #     print()
    
# # #     # Copy missing files
# # #     copy_missing_files(source_dir, target_dir)

# # # if __name__ == "__main__":
# # #     main()

import os
import csv
from pathlib import Path

def process_summary_files():
    # Base folders
    # base_folders = ['pruned_ModeA', 'pruned_ModeB', 'pruned_ModeC', 'pruned_ModeD']
    base_folders = ['pruned_baselines/delaunay_k', 'pruned_baselines/nearest_k', 'pruned_baselines/random_k']
    k_values = ['k5', 'k10', 'k15', 'k20', 'k25']
    
    # Iterate through all folders and subfolders
    for base_folder in base_folders:
        for k_folder in k_values:
            # Construct the file path
            file_path = Path(base_folder) / k_folder / f'summary_{k_folder}.csv'
            
            # Check if file exists
            if not file_path.exists():
                print(f"Warning: {file_path} not found")
                continue
            
            # Read the CSV file
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                # Get headers from first row
                headers = rows[0]
                
                # Find the average row (starts with "** AVERAGE **")
                avg_row = None
                for row in rows[1:]:  # Skip header row
                    if row and row[0].strip() == '** AVERAGE **':
                        avg_row = row
                        break
                
                if not avg_row:
                    print(f"Warning: No average row found in {file_path}")
                    continue
                
                # Print the file path
                print(f"\n{'='*60}")
                print(f"{file_path}:")
                print(f"{'-'*40}")
                
                # Print each variable with its average value
                # Skip the first column which contains "** AVERAGE **" text
                for i in range(1, len(headers)):
                    if i < len(avg_row):
                        var_name = headers[i].strip()
                        var_value = avg_row[i].strip()
                        print(f"{var_name}: {var_value}")
                
                print(f"{'='*60}")

if __name__ == "__main__":
    process_summary_files()

# import os, sys
# import torch
# import numpy as np
# from sklearn.model_selection import train_test_split

# # Same imports and functions from your training code
# from preprocess_data import build_pyg_data_from_instance
# from newTrain import collect_pairs

# def load_or_create_data(args):
#     cache_path = "data_cache/pyg_graphs.pt"
#     if os.path.exists(cache_path):
#         return torch.load(cache_path, weights_only=False)
#     # else build from scratch (reuse your existing function)

# # Simulate args with same dirs you used for training
# class Args:
#     train_dir = "tsplib_data"
#     synthetic_dir = "synthetic_tsplib"
#     full_threshold = 300
#     knn_k = 30
#     knn_feat_k = 10

# args = Args()
# data_list = load_or_create_data(args)
# print(f"Loaded {len(data_list)} graphs")

# # Split with the chosen seed (42)
# idxs = list(range(len(data_list)))
# train_idx, test_idx = train_test_split(idxs, test_size=0.2, random_state=42)
# test_names = [data_list[i].name for i in test_idx]
# print(f"Test set has {len(test_names)} instances")
# # Save names
# with open("test_instance_names_seed42.txt", "w") as f:
#     f.write("\n".join(test_names))




# import os
# import shutil
# from pathlib import Path

# def copy_instance_files(name, src_dir, dest_dir):
#     """Copy name.tsp and name.opt.tour from src_dir to dest_dir."""
#     tsp_src = src_dir / f"{name}.tsp"
#     tour_src = src_dir / f"{name}.opt.tour"
    
#     if tsp_src.exists() and tour_src.exists():
#         shutil.copy2(tsp_src, dest_dir)
#         shutil.copy2(tour_src, dest_dir)
#         print(f"Copied {name} from {src_dir}")
#         return True
#     return False

# def main():
#     names_file = "test_instance_names_seed42.txt"
#     dest_dir = Path("graphss_chosen")
#     dest_dir.mkdir(exist_ok=True)
    
#     search_dirs = [Path("synthetic_tsplib"), Path("tsplib_data")]
    
#     with open(names_file, "r") as f:
#         for line in f:
#             name = line.strip()
#             if not name:  # skip empty lines
#                 continue
            
#             copied = False
#             for src_dir in search_dirs:
#                 if copy_instance_files(name, src_dir, dest_dir):
#                     copied = True
#                     break
            
#             if not copied:
#                 print(f"Skipped {name}: files not found in any search directory")

# if __name__ == "__main__":
#     main()