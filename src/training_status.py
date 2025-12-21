"""
Script to check the status of all training combinations.
"""
import json
import fcntl
import time
from pathlib import Path
from datetime import datetime


STATUS_FILE = Path("results/training_status.json")
STATUS_LOCK_FILE = Path("results/training_status.lock")


def acquire_status_lock(lock_file_handle, timeout=30):
    """Acquire an exclusive lock on the status file."""
    start_time = time.time()
    while True:
        try:
            fcntl.flock(lock_file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except IOError:
            if time.time() - start_time > timeout:
                return False
            time.sleep(0.1)


def release_status_lock(lock_file_handle):
    """Release the lock on the status file."""
    fcntl.flock(lock_file_handle, fcntl.LOCK_UN)


def read_status_file():
    """Safely read the status file with locking."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Create status file if it doesn't exist
    if not STATUS_FILE.exists():
        return None

    with open(STATUS_LOCK_FILE, 'a') as lock_file:
        if not acquire_status_lock(lock_file):
            raise TimeoutError("Could not acquire status file lock")

        try:
            with open(STATUS_FILE, 'r') as f:
                content = f.read().strip()
                if not content:
                    return None
                return json.loads(content)
        finally:
            release_status_lock(lock_file)


def print_detailed_status():
    """Print detailed status of all training combinations."""
    status = read_status_file()

    if status is None:
        print("No training status file found.")
        print("Run 'make train-several' to initialize training.")
        return

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
