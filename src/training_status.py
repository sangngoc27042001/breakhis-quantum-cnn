"""
Script to check the status of all training combinations.
"""
import json
from pathlib import Path
from datetime import datetime


STATUS_FILE = Path("results/training_status.json")


def print_detailed_status():
    """Print detailed status of all training combinations."""
    if not STATUS_FILE.exists():
        print("No training status file found.")
        print("Run 'make train-several' to initialize training.")
        return

    with open(STATUS_FILE, 'r') as f:
        status = json.load(f)

    combinations = status["combinations"]

    # Count by status
    pending = [k for k, v in combinations.items() if v["status"] == "pending"]
    in_progress = [k for k, v in combinations.items() if v["status"] == "in_progress"]
    completed = [k for k, v in combinations.items() if v["status"] == "completed"]
    failed = [k for k, v in combinations.items() if v["status"] == "failed"]

    total = len(combinations)

    print("\n" + "=" * 80)
    print("TRAINING STATUS - DETAILED VIEW")
    print("=" * 80)
    print(f"Total combinations: {total}")
    print(f"Pending:            {len(pending)}")
    print(f"In Progress:        {len(in_progress)}")
    print(f"Completed:          {len(completed)}")
    print(f"Failed:             {len(failed)}")
    print("=" * 80)

    if in_progress:
        print("\n--- IN PROGRESS ---")
        for combo_id in sorted(in_progress):
            info = combinations[combo_id]
            config = info["config"]
            print(f"\n{combo_id}:")
            print(f"  Config: qubits={config['n_qubits']}, template={config['template']}, depth={config['depth']}")
            print(f"  Host: {info['hostname']}, PID: {info['pid']}")
            print(f"  Started: {info['started_at']}")

    if completed:
        print("\n--- COMPLETED ---")
        for combo_id in sorted(completed):
            info = combinations[combo_id]
            config = info["config"]
            print(f"{combo_id}: qubits={config['n_qubits']}, template={config['template']}, depth={config['depth']}")

    if failed:
        print("\n--- FAILED ---")
        for combo_id in sorted(failed):
            info = combinations[combo_id]
            config = info["config"]
            print(f"{combo_id}: qubits={config['n_qubits']}, template={config['template']}, depth={config['depth']}")

    if pending:
        print("\n--- PENDING ---")
        for combo_id in sorted(pending):
            info = combinations[combo_id]
            config = info["config"]
            print(f"{combo_id}: qubits={config['n_qubits']}, template={config['template']}, depth={config['depth']}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    print_detailed_status()
