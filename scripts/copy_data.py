#!/usr/bin/env python3
"""
Script to copy the data directory to build/ for static site generation.

This script copies the entire data directory structure to build/data,
preserving all subdirectories and files for use in static site builds.

Usage:
    python scripts/copy_data.py
    uv run python scripts/copy_data.py
"""

import shutil
import sys
from pathlib import Path


def copy_data_to_build():
    """Copy the data directory to build/data."""
    # Define source and destination paths
    project_root = Path(__file__).parent.parent
    source_dir = project_root / "data"
    build_dir = project_root / "build"
    dest_dir = build_dir / "data"

    # Check if source directory exists
    if not source_dir.exists():
        print(f"❌ Error: Source directory '{source_dir}' does not exist")
        return False

    # Create build directory if it doesn't exist
    build_dir.mkdir(exist_ok=True)

    # Remove existing data directory in build if it exists
    if dest_dir.exists():
        print(f"🗑️  Removing existing '{dest_dir}'...")
        shutil.rmtree(dest_dir)

    try:
        # Copy the entire data directory
        print(f"📁 Copying '{source_dir}' to '{dest_dir}'...")
        shutil.copytree(source_dir, dest_dir)

        # Count copied items for feedback
        total_files = sum(1 for _ in dest_dir.rglob("*") if _.is_file())
        total_dirs = sum(1 for _ in dest_dir.rglob("*") if _.is_dir())

        print(f"✅ Successfully copied data directory!")
        print(f"   📄 Files: {total_files}")
        print(f"   📁 Directories: {total_dirs}")
        print(f"   📍 Destination: {dest_dir}")

        return True

    except Exception as e:
        print(f"❌ Error copying data directory: {e}")
        return False


def main():
    """Main function to run the data copy operation."""
    print("🚀 Starting data copy operation...")
    print("=" * 50)

    success = copy_data_to_build()

    print("=" * 50)
    if success:
        print("🎉 Data copy completed successfully!")
        sys.exit(0)
    else:
        print("💥 Data copy failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
