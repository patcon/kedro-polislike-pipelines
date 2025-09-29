#!/usr/bin/env python3
"""
Script to replace absolute file paths with relative paths in API response files.

This script finds all instances of '/Users/patcon/repos/kedro-polis-pipelines/'
in JSON files within the build/api/ directory and replaces them with empty string
to make the paths relative.

Example:
  Before: "/Users/patcon/repos/kedro-polis-pipelines/data/6carwc4nzj/knn5d_pca_bestkmeans/03_primary/projections.json"
  After:  "data/6carwc4nzj/knn5d_pca_bestkmeans/03_primary/projections.json"
"""

import os
import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union


def find_and_replace_paths_in_dict(
    data: Union[Dict[str, Any], List[Any]], old_path: str, new_path: str = ""
) -> Tuple[Union[Dict[str, Any], List[Any]], int]:
    """
    Recursively find and replace path strings in a dictionary or list.

    Args:
        data: Dictionary or list to process
        old_path: Path string to replace
        new_path: Replacement string (default: empty string)

    Returns:
        Tuple of (modified_data, replacement_count)
    """
    replacement_count = 0

    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if isinstance(value, str) and old_path in value:
                result[key] = value.replace(old_path, new_path)
                replacement_count += 1
            elif isinstance(value, (dict, list)):
                result[key], sub_count = find_and_replace_paths_in_dict(
                    value, old_path, new_path
                )
                replacement_count += sub_count
            else:
                result[key] = value
        return result, replacement_count

    elif isinstance(data, list):
        result = []
        for item in data:
            if isinstance(item, str) and old_path in item:
                result.append(item.replace(old_path, new_path))
                replacement_count += 1
            elif isinstance(item, (dict, list)):
                modified_item, sub_count = find_and_replace_paths_in_dict(
                    item, old_path, new_path
                )
                result.append(modified_item)
                replacement_count += sub_count
            else:
                result.append(item)
        return result, replacement_count

    else:
        return data, replacement_count


def process_api_file(
    file_path: Path, old_path: str, new_path: str = ""
) -> Tuple[bool, int]:
    """
    Process a single API file to replace paths.

    Args:
        file_path: Path to the file to process
        old_path: Path string to replace
        new_path: Replacement string (default: empty string)

    Returns:
        Tuple of (success, replacement_count)
    """
    try:
        # Read the file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Try to parse as JSON
        try:
            data = json.loads(content)
            modified_data, replacement_count = find_and_replace_paths_in_dict(
                data, old_path, new_path
            )

            if replacement_count > 0:
                # Write back the modified JSON
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(modified_data, f, indent=2, ensure_ascii=False)
                print(f"✓ {file_path}: {replacement_count} replacements")
            else:
                print(f"- {file_path}: no changes needed")

            return True, replacement_count

        except json.JSONDecodeError:
            # If it's not valid JSON, try simple string replacement
            if old_path in content:
                modified_content = content.replace(old_path, new_path)
                replacement_count = content.count(old_path)

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(modified_content)
                print(f"✓ {file_path}: {replacement_count} replacements (text mode)")
                return True, replacement_count
            else:
                print(f"- {file_path}: no changes needed (text mode)")
                return True, 0

    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False, 0


def find_api_files(api_dir: Path) -> List[Path]:
    """
    Find all files in the API directory (build/api/ only).

    Args:
        api_dir: Path to the API directory

    Returns:
        List of file paths to process
    """
    files = []

    # Only process files in build/api/ directory, not data files
    for root, dirs, filenames in os.walk(api_dir):
        for filename in filenames:
            file_path = Path(root) / filename
            # Skip hidden files and directories
            if not any(part.startswith(".") for part in file_path.parts):
                files.append(file_path)

    return sorted(files)


def main():
    """Main function to process all API files."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Replace absolute paths with relative paths in API files"
    )
    parser.add_argument(
        "--old-path",
        type=str,
        help="Absolute path to replace (default: current working directory + '/')",
    )
    parser.add_argument(
        "--new-path",
        type=str,
        default="",
        help="Replacement path (default: empty string)",
    )
    parser.add_argument(
        "--api-dir",
        type=str,
        default="build/api",
        help="API directory to process (default: build/api)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes",
    )

    args = parser.parse_args()

    # Configuration
    api_dir = Path(args.api_dir)

    # Determine the old path dynamically if not provided
    if args.old_path:
        old_path = args.old_path
    else:
        # Use current working directory + trailing slash
        old_path = str(Path.cwd()) + "/"

    new_path = args.new_path

    # Check if API directory exists
    if not api_dir.exists():
        print(f"Error: API directory '{api_dir}' does not exist.")
        return 1

    print(f"Processing API files in '{api_dir}'...")
    print(
        f"Replacing: '{old_path}' -> '{new_path}' {'(empty string for relative paths)' if new_path == '' else ''}"
    )
    if args.dry_run:
        print("DRY RUN MODE: No files will be modified")
    print("-" * 70)

    # Find all files to process
    files_to_process = find_api_files(api_dir)

    if not files_to_process:
        print("No files found to process.")
        return 0

    print(f"Found {len(files_to_process)} files to process.\n")

    # Process each file
    total_files_processed = 0
    total_files_modified = 0
    total_replacements = 0
    failed_files = []

    for file_path in files_to_process:
        if args.dry_run:
            # In dry run mode, just check for matches without modifying
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if old_path in content:
                    replacement_count = content.count(old_path)
                    print(f"Would modify {file_path}: {replacement_count} replacements")
                    total_files_modified += 1
                    total_replacements += replacement_count
                else:
                    print(f"- {file_path}: no changes needed")

                total_files_processed += 1
            except Exception as e:
                print(f"✗ Error reading {file_path}: {e}")
                failed_files.append(file_path)
        else:
            success, replacement_count = process_api_file(file_path, old_path, new_path)

            if success:
                total_files_processed += 1
                if replacement_count > 0:
                    total_files_modified += 1
                    total_replacements += replacement_count
            else:
                failed_files.append(file_path)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"Files processed: {total_files_processed}/{len(files_to_process)}")
    print(f"Files modified: {total_files_modified}")
    print(f"Total replacements: {total_replacements}")

    if failed_files:
        print(f"Failed files: {len(failed_files)}")
        for failed_file in failed_files:
            print(f"  - {failed_file}")
        return 1
    else:
        print("All files processed successfully!")
        return 0


if __name__ == "__main__":
    exit(main())
