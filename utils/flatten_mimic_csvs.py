"""
Utility to flatten nested MIMIC CSV directories.
Moves files like hosp/admissions.csv/admissions.csv to hosp/admissions.csv
Removes empty subdirectories after moving files.

Usage: Run this script from the project root or directly.
"""
import os
import shutil
from pathlib import Path
import logging

def flatten_nested_csv_dirs(base_dir):
    """
    For each subdirectory in base_dir, if it contains a file with the same name as the subdirectory,
    move the file up one level and remove the empty subdirectory.
    """
    base = Path(base_dir)
    # First, rename all subdirectories ending with .csv to remove the .csv extension
    for subdir in list(base.iterdir()):
        if subdir.is_dir() and subdir.name.endswith('.csv'):
            new_name = subdir.name[:-4]  # Remove .csv
            new_dir = base / new_name
            if new_dir.exists():
                logging.warning(f"Target directory {new_dir} already exists. Skipping rename for {subdir}.")
                continue
            logging.info(f"Renaming directory {subdir} -> {new_dir}")
            subdir.rename(new_dir)

    # Now, flatten the structure as before
    for subdir in base.iterdir():
        if subdir.is_dir():
            expected_file = subdir / (subdir.name + '.csv')
            target_file = base / (subdir.name + '.csv')
            if expected_file.exists():
                if target_file.exists():
                    logging.warning(f"Target file {target_file} already exists. Skipping move for {expected_file}.")
                    continue
                logging.info(f"Moving {expected_file} -> {target_file}")
                shutil.move(str(expected_file), str(target_file))
                # Remove the now-empty directory
                try:
                    subdir.rmdir()
                    logging.info(f"Removed empty directory {subdir}")
                except OSError:
                    logging.warning(f"Could not remove {subdir}, not empty.")

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Flatten nested MIMIC CSV directories.")
    parser.add_argument("--base", type=str, default="src/data/mimic/hosp", help="Base directory to flatten (default: src/data/mimic/hosp)")
    parser.add_argument("--also-icu", action="store_true", help="Also flatten src/data/mimic/icu")
    args = parser.parse_args()

    flatten_nested_csv_dirs(args.base)
    if args.also_icu:
        flatten_nested_csv_dirs("src/data/mimic/icu")
    print("Flattening complete.")
