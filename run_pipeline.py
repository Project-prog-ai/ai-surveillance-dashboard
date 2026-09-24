"""
run_pipeline.py — Full Reproducibility Pipeline
================================================

Runs the complete evaluation pipeline and generates all results.

Usage:
    python run_pipeline.py

This will:
    1. Validate the dataset
    2. Run all model evaluations
    3. Run tests
    4. Generate results/ CSV files
    5. Print summary
"""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

def run_cmd(label, cmd):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    if result.returncode != 0:
        print(f"  WARNING: {label} returned code {result.returncode}")
    return result.returncode

def main():
    print("AI Vehicle Threat Assessment — Full Pipeline")
    print("=" * 60)

    # Step 1: Run evaluation suite
    rc1 = run_cmd(
        "Step 1: Model Evaluation",
        f"{sys.executable} evaluate_models.py"
    )

    # Step 2: Run tests
    rc2 = run_cmd(
        "Step 2: Automated Tests",
        f"{sys.executable} -m pytest tests/ -v --tb=short"
    )

    # Step 3: Summary
    print(f"\n{'='*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  Evaluation: {'PASS' if rc1 == 0 else 'FAIL'}")
    print(f"  Tests:      {'PASS' if rc2 == 0 else 'FAIL'}")

    results_dir = os.path.join(ROOT, "results")
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        print(f"\n  Generated result files ({len(files)}):")
        for f in sorted(files):
            size = os.path.getsize(os.path.join(results_dir, f))
            print(f"    - {f} ({size} bytes)")

    print(f"\nDone. Results in: {results_dir}")

if __name__ == "__main__":
    main()
