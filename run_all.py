"""
MASTER PIPELINE RUNNER
=========================
Runs Stages 1-10 in sequence. Stage 1 (config.py) is imported by every
other stage automatically, so it is not "run" standalone.

Usage: python3 run_all.py
"""
import subprocess
import sys
import time

STAGES = [
    "stage2_generate_dataset.py",
    "stage3_preprocess_features.py",
    "stage4_api_gateway_simulator.py",
    "stage5_transaction_switch_simulator.py",
    "stage6_ml_intelligent_router.py",
    "stage7_benchmark_runner.py",
    "stage8_statistical_validation.py",
    "stage9_visualization.py",
    "stage10_final_report.py",
]

if __name__ == "__main__":
    t0 = time.time()
    for i, stage in enumerate(STAGES, start=2):
        print(f"\n{'='*70}\nRunning Stage {i}: {stage}\n{'='*70}")
        result = subprocess.run([sys.executable, stage])
        if result.returncode != 0:
            print(f"Pipeline halted: {stage} exited with code {result.returncode}")
            sys.exit(1)
    print(f"\nPipeline complete in {time.time()-t0:.1f}s — see /home/claude/project/results")
