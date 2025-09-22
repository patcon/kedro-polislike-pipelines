#!/usr/bin/env python3
"""
Script to run Kedro pipelines with filtering and visualization options.

Usage:
    uv run python scripts/run_pipelines.py --all [--launch-viz]
    uv run python scripts/run_pipelines.py pipeline1,pipeline2 [--launch-viz]
    uv run python scripts/run_pipelines.py pattern [--params "key1=value1,key2=value2"]
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
        from kedro_polis_classic.pipeline_registry import register_pipelines

        pipelines = register_pipelines()
        # Filter out __default__ as it's typically an alias
        available = {name for name in pipelines.keys() if name != "__default__"}
        return available

    except Exception as e:
        print(f"Error getting available pipelines: {e}")
        print("Falling back to configuration-based discovery...")

        # Fallback: read from configuration
        try:
            config_loader = OmegaConfigLoader(
                conf_source="conf", base_env="base", default_run_env="local"
            )
            params = config_loader["parameters"]
            experimental_names = set(params.get("pipelines", {}).keys())
            # Add known base pipelines
            base_pipelines = {"polis", "polis_classic"}
            return base_pipelines.union(experimental_names)
        except Exception as fallback_e:
            print(f"Fallback also failed: {fallback_e}")
            return set()


def filter_pipelines(requested: List[str], available: Set[str]) -> List[str]:
    """Filter requested pipelines against available ones, supporting pattern matching."""
    if not available:
        print(
            "Warning: Could not determine available pipelines. Proceeding with requested pipelines."
        )
        return requested

    valid_pipelines = []
    unmatched_patterns = []

    for pattern in requested:
        # First try exact match
        if pattern in available:
            valid_pipelines.append(pattern)
        else:
            # Try pattern matching (substring match)
            matches = [p for p in available if pattern in p]
            if matches:
                valid_pipelines.extend(matches)
                print(f"Pattern '{pattern}' matched: {', '.join(sorted(matches))}")
            else:
                unmatched_patterns.append(pattern)

    if unmatched_patterns:
        print(
            f"Warning: The following patterns had no matches: {', '.join(unmatched_patterns)}"
        )
        print(f"Available pipelines: {', '.join(sorted(available))}")

    # Remove duplicates while preserving order
    seen = set()
    unique_pipelines = []
    for pipeline in valid_pipelines:
        if pipeline not in seen:
            seen.add(pipeline)
            unique_pipelines.append(pipeline)

    return unique_pipelines


def run_pipeline(
    pipeline_name: str, current: int, total: int, params: str | None = None
) -> bool:
    """Run a single pipeline using kedro run command."""
    print(f"\n{'=' * 60}")
    print(f"Running pipeline ({current}/{total}): {pipeline_name}")
    if params:
        print(f"With parameters: {params}")
    print(f"{'=' * 60}")

    cmd = ["kedro", "run", "--pipeline", pipeline_name]
    if params:
        cmd.extend(["--params", params])

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
        description="Run Kedro pipelines with filtering and visualization options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/run_pipelines.py --all
  uv run python scripts/run_pipelines.py --all --launch-viz
  uv run python scripts/run_pipelines.py polis,mean_pca_bestkmeans
  uv run python scripts/run_pipelines.py mean_pacmap_kmeans --launch-viz
  uv run python scripts/run_pipelines.py besthdbscanflat --params "param1=value1,param2=value2"
        """,
    )

    # Pipeline selection arguments (mutually exclusive)
    pipeline_group = parser.add_mutually_exclusive_group(required=True)
    pipeline_group.add_argument(
        "--all", action="store_true", help="Run all available pipelines"
    )
    pipeline_group.add_argument(
        "pipelines", nargs="?", help="Comma-separated list of pipeline names to run"
    )

    # Optional flags
    parser.add_argument(
        "--launch-viz",
        action="store_true",
        help="Launch Kedro Viz after running pipelines",
    )
    parser.add_argument(
        "--params",
        type=str,
        help="Parameters to pass to kedro run (e.g., 'param1:value1,param2:value2')",
    )

    args = parser.parse_args()

    # Get available pipelines
    print("Discovering available pipelines...")
    available_pipelines = get_available_pipelines()

    if available_pipelines:
        print(
            f"Found {len(available_pipelines)} available pipelines: {', '.join(sorted(available_pipelines))}"
        )
    else:
        print("Could not discover available pipelines automatically.")

    # Determine which pipelines to run
    if args.all:
        if not available_pipelines:
            print("❌ Error: Cannot run all pipelines - pipeline discovery failed.")
            sys.exit(1)
        pipelines_to_run = sorted(available_pipelines)
        print(f"\nRunning ALL pipelines: {', '.join(pipelines_to_run)}")
    else:
        requested_pipelines = [p.strip() for p in args.pipelines.split(",")]
        pipelines_to_run = filter_pipelines(requested_pipelines, available_pipelines)

        if not pipelines_to_run:
            print("❌ Error: No valid pipelines to run.")
            sys.exit(1)

        print(f"\nRunning selected pipelines: {', '.join(pipelines_to_run)}")

    # Run pipelines
    successful_runs = 0
    failed_runs = 0
    failed_pipelines = []

    total_pipelines = len(pipelines_to_run)
    for i, pipeline in enumerate(pipelines_to_run, 1):
        success = run_pipeline(pipeline, i, total_pipelines, args.params)
        if success:
            successful_runs += 1
        else:
            failed_runs += 1
            failed_pipelines.append(pipeline)

    # Summary
    print(f"\n{'=' * 60}")
    print("EXECUTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"✅ Successful: {successful_runs}")
    print(f"❌ Failed: {failed_runs}")
    print(f"📊 Total: {successful_runs + failed_runs}")

    if failed_pipelines:
        print(f"\n❌ Failed pipelines:")
        for pipeline in failed_pipelines:
            print(f"   • {pipeline}")

    # Launch viz if requested
    if args.launch_viz:
        launch_viz()

    # Exit with appropriate code
    if failed_runs > 0:
        sys.exit(1)
    else:
        print("\n🎉 All pipelines completed successfully!")


if __name__ == "__main__":
    main()
