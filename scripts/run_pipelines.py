#!/usr/bin/env python3
"""
Script to run the default Kedro pipeline with tag filtering and visualization options.

Usage:
    uv run python scripts/run_pipelines.py [--tags "tag1,tag2"] [--launch-viz]
    uv run python scripts/run_pipelines.py --tags "pca,bestkmeans" [--params "key1=value1,key2=value2"]
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Set

from kedro.config import OmegaConfigLoader
from kedro.framework.project import configure_project
from kedro.framework.startup import bootstrap_project


def get_available_pipelines() -> Set[str]:
    """Get all available pipeline names from the project."""
    try:
        # Bootstrap the Kedro project
        project_path = Path(__file__).parent.parent
        bootstrap_project(project_path)

        # Import and get pipelines
        from kedro_polislike_pipelines.pipeline_registry import register_pipelines

        pipelines = register_pipelines()
        # Return the default pipeline name
        return {"__default__"}

    except Exception as e:
        print(f"Error getting available pipelines: {e}")
        print("Falling back to configuration-based discovery...")

        # Fallback: return default pipeline
        return {"__default__"}


def get_default_pipeline() -> str:
    """Get the default pipeline name."""
    return "__default__"


def run_pipeline(
    pipeline_name: str, params: str | None = None, tags: str | None = None
) -> bool:
    """Run the default pipeline using kedro run command."""
    print(f"\n{'=' * 60}")
    print(f"Running pipeline: {pipeline_name}")
    if params:
        print(f"With parameters: {params}")
    if tags:
        print(f"With tags: {tags}")
    print(f"{'=' * 60}")

    cmd = ["kedro", "run"]
    if params:
        cmd.extend(["--params", params])
    if tags:
        cmd.extend(["--tags", tags])

    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent)
        print(f"✅ Pipeline '{pipeline_name}' completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline '{pipeline_name}' failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(
            "❌ Error: 'kedro' command not found. Make sure Kedro is installed and in your PATH."
        )
        return False


def launch_viz() -> None:
    """Launch Kedro Viz and keep it running."""
    print(f"\n{'=' * 60}")
    print("Launching Kedro Viz...")
    print("Press Ctrl+C to stop Kedro Viz and exit")
    print(f"{'=' * 60}")

    cmd = ["kedro", "viz", "--autoreload"]

    try:
        # Use subprocess.run to keep the process running in foreground
        print("🚀 Starting Kedro Viz... Check your browser at http://localhost:4141")
        subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    except KeyboardInterrupt:
        print("\n🛑 Kedro Viz stopped by user")
    except FileNotFoundError:
        print(
            "❌ Error: 'kedro' command not found. Make sure Kedro is installed and in your PATH."
        )
    except Exception as e:
        print(f"❌ Error launching Kedro Viz: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the default Kedro pipeline with tag filtering and visualization options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/run_pipelines.py
  uv run python scripts/run_pipelines.py --launch-viz
  uv run python scripts/run_pipelines.py --tags "pca,bestkmeans"
  uv run python scripts/run_pipelines.py --tags "pca_bestkmeans,umap_bestkmeans" --launch-viz
  uv run python scripts/run_pipelines.py --tags "bestkmeans" --params "param1=value1,param2=value2"
        """,
    )

    # Optional flags
    parser.add_argument(
        "--launch-viz",
        action="store_true",
        help="Launch Kedro Viz after running the pipeline",
    )
    parser.add_argument(
        "--params",
        type=str,
        help="Parameters to pass to kedro run (e.g., 'param1:value1,param2:value2')",
    )
    parser.add_argument(
        "--tags",
        type=str,
        help="Comma-separated tags to filter nodes in the pipeline (e.g., 'pca_bestkmeans,umap_bestkmeans' or 'pca,bestkmeans')",
    )

    args = parser.parse_args()

    # Get the default pipeline
    pipeline_name = get_default_pipeline()
    print(f"Running default pipeline: {pipeline_name}")
    
    if args.tags:
        print(f"With tags: {args.tags}")
    if args.params:
        print(f"With parameters: {args.params}")

    # Run the pipeline
    success = run_pipeline(pipeline_name, args.params, args.tags)

    # Summary
    print(f"\n{'=' * 60}")
    print("EXECUTION SUMMARY")
    print(f"{'=' * 60}")
    if success:
        print("✅ Pipeline completed successfully!")
    else:
        print("❌ Pipeline failed!")

    # Launch viz if requested
    if args.launch_viz:
        launch_viz()

    # Exit with appropriate code
    if not success:
        sys.exit(1)
    else:
        print("\n🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    main()
