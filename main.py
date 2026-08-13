#!/usr/bin/env python3
"""
Path Planning Algorithm - Main Entry Point

Usage:
    python main.py [config_path] [output_root] [run_name]

Default config: Algorithm description/路径规划算法_工程运行参数.toml
"""
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.launcher import run_pipeline


def main():
    t0 = time.time()
    print("=" * 70)
    print("Path Planning Algorithm - Main Entry")
    print("=" * 70)

    # Default config path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(
        script_dir, "Algorithm description",
        "路径规划算法_工程运行参数.toml"
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else default_config
    output_root = sys.argv[2] if len(sys.argv) > 2 else "./outputs"
    run_name = sys.argv[3] if len(sys.argv) > 3 else ""

    # Verify config exists
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Config: {config_path}")
    print(f"Output root: {output_root}")
    print(f"Run name: {run_name or '(auto)'}")
    print()

    # Run pipeline
    result = run_pipeline(
        config_path=config_path,
        output_root=output_root,
        run_name=run_name,
        overwrite_policy="reject",
    )

    # Print summary
    total_time = time.time() - t0
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Status: {result.status}")
    print(f"Completed stages: {result.completed_stages}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Output: {result.output_directory}")

    if result.final_result and result.final_result.final_metrics:
        m = result.final_result.final_metrics
        print(f"J_true = {m.get('J_true', 'N/A'):.4f}")
        print(f"  L = {m.get('L', 'N/A'):.2f}")
        print(f"  D = {m.get('D', 'N/A'):.2f}")
        print(f"  A = {m.get('A', 'N/A'):.2f}")
        print(f"  R = {m.get('R', 'N/A'):.2f}")

    if result.failed_stage > 0:
        print(f"Failed at stage {result.failed_stage}: {result.failure_reason}")
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()