"""
Script to train multiple backbone models with fixed quantum configuration.
Supports running multiple terminals simultaneously without conflicts.
"""
import json
import os
import time
import socket
import fcntl
from datetime import datetime
from pathlib import Path

import torch
from src import config
from src.train import main


LOCK_DIR = Path("results/training_locks_backbones")
STATUS_FILE = Path("results/training_status_backbones.json")
STATUS_LOCK_FILE = Path("results/training_status_backbones.lock")


# Available small models to rotate through
AVAILABLE_SMALL_MODELS = [
    # "mobilenetv3_small_100",  # 1.53M params, 96.5ms
    # "mnasnet_100",            # 3.11M params, 226.52ms
    "regnetx_002",            # 2.32M params, 24.35ms (second fastest)
    "regnety_002",            # 2.80M params, 23.33ms (fastest)
    # "ghostnet_100",           # 3.91M params, 62.38ms (largest small model)
    # "efficientnet_lite0",     # 3.38M params, 285.4ms
    "mobilevit_xs",           # 1.94M params, 70.36ms (hybrid CNN-Transformer)
]


def clear_gpu_memory():
    """Clear GPU cache and force garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("GPU memory cleared")


def get_combination_id(model_config):
    """Generate unique ID for a backbone model."""
    return f"backbone_{model_config['backbone']}"


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
        with open(STATUS_FILE, 'w') as f:
            json.dump({"combinations": {}}, f, indent=2)

    with open(STATUS_LOCK_FILE, 'a') as lock_file:
        if not acquire_status_lock(lock_file):
            raise TimeoutError("Could not acquire status file lock")

        try:
            with open(STATUS_FILE, 'r') as f:
                content = f.read().strip()
                if not content:
                    return {"combinations": {}}
                return json.loads(content)
        finally:
            release_status_lock(lock_file)


def write_status_file(status):
    """Safely write the status file with locking."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(STATUS_LOCK_FILE, 'a') as lock_file:
        if not acquire_status_lock(lock_file):
            raise TimeoutError("Could not acquire status file lock")

        try:
            with open(STATUS_FILE, 'w') as f:
                json.dump(status, f, indent=2)
        finally:
            release_status_lock(lock_file)


def initialize_status_file(all_combinations):
    """Initialize or update the status file with all combinations."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)

    status = read_status_file()

    # Add any new combinations
    for config in all_combinations:
        combo_id = get_combination_id(config)
        if combo_id not in status["combinations"]:
            status["combinations"][combo_id] = {
                "config": config,
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "hostname": None,
                "pid": None
            }

    write_status_file(status)
    return status


def claim_combination(combo_id):
    """
    Attempt to claim a combination for training.
    Returns True if successfully claimed, False if already claimed by someone else.
    """
    lock_file = LOCK_DIR / f"{combo_id}.lock"

    # Try to create lock file exclusively
    try:
        # Check if lock file exists and is stale (older than 24 hours)
        if lock_file.exists():
            lock_age = time.time() - lock_file.stat().st_mtime
            if lock_age > 86400:  # 24 hours
                print(f"Removing stale lock file for {combo_id} (age: {lock_age/3600:.1f} hours)")
                lock_file.unlink()

        # Try to create the lock file atomically
        lock_info = {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "claimed_at": datetime.now().isoformat()
        }

        # Write to a temporary file first, then rename atomically
        temp_lock = lock_file.with_suffix('.lock.tmp')
        with open(temp_lock, 'w') as f:
            json.dump(lock_info, f, indent=2)

        # Atomic rename - will fail if lock file already exists
        try:
            os.link(temp_lock, lock_file)
            temp_lock.unlink()
        except FileExistsError:
            temp_lock.unlink()
            raise

        # Update status file safely
        status = read_status_file()
        status["combinations"][combo_id]["status"] = "in_progress"
        status["combinations"][combo_id]["started_at"] = datetime.now().isoformat()
        status["combinations"][combo_id]["hostname"] = socket.gethostname()
        status["combinations"][combo_id]["pid"] = os.getpid()
        write_status_file(status)

        return True

    except FileExistsError:
        # Lock already exists - try to read it safely
        try:
            with open(lock_file, 'r') as f:
                lock_info = json.load(f)
            print(f"Combination {combo_id} is already claimed by {lock_info['hostname']} (PID: {lock_info['pid']})")
        except (json.JSONDecodeError, FileNotFoundError):
            # Lock file is corrupted or disappeared, try again
            print(f"Combination {combo_id} has a corrupted lock file, skipping...")
        return False


def release_combination(combo_id, success=True):
    """Release the lock on a combination and update its status."""
    lock_file = LOCK_DIR / f"{combo_id}.lock"

    # Remove lock file
    if lock_file.exists():
        lock_file.unlink()

    # Update status file safely
    status = read_status_file()
    status["combinations"][combo_id]["status"] = "completed" if success else "failed"
    status["combinations"][combo_id]["completed_at"] = datetime.now().isoformat()
    write_status_file(status)


def get_next_available_combination():
    """
    Find and claim the next available combination.
    Returns the combination config or None if all are taken/completed.
    """
    status = read_status_file()

    for combo_id, combo_info in status["combinations"].items():
        if combo_info["status"] == "pending":
            if claim_combination(combo_id):
                return combo_info["config"], combo_id

    return None, None


def print_training_status():
    """Print the current status of all training combinations."""
    if not STATUS_FILE.exists():
        print("No training status file found.")
        return

    status = read_status_file()

    pending = sum(1 for c in status["combinations"].values() if c["status"] == "pending")
    in_progress = sum(1 for c in status["combinations"].values() if c["status"] == "in_progress")
    completed = sum(1 for c in status["combinations"].values() if c["status"] == "completed")
    failed = sum(1 for c in status["combinations"].values() if c["status"] == "failed")
    total = len(status["combinations"])

    print("\n" + "=" * 80)
    print("BACKBONE TRAINING STATUS SUMMARY")
    print("=" * 80)
    print(f"Total backbones:    {total}")
    print(f"Pending:            {pending}")
    print(f"In Progress:        {in_progress}")
    print(f"Completed:          {completed}")
    print(f"Failed:             {failed}")
    print("=" * 80)

    if in_progress > 0:
        print("\nCurrently training:")
        for combo_id, combo_info in status["combinations"].items():
            if combo_info["status"] == "in_progress":
                config = combo_info["config"]
                print(f"  {combo_id}: backbone={config['backbone']}")
                print(f"    Host: {combo_info['hostname']}, PID: {combo_info['pid']}")
                print(f"    Started: {combo_info['started_at']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    """
    Train quantum CNN models with different backbones and fixed quantum configuration.
    Multiple terminals can run this simultaneously - each will claim and train
    different backbones automatically.

    Fixed quantum configuration:
    - Qubits: 8
    - Template: two_design
    - Depth: 5

    Total: 7 backbone models
    """
    # Base configuration
    base_model = "cnn_quantum"
    encoding_method = "rotation"

    # Fixed quantum parameters
    n_qubits = 8
    template = "two_design"
    depth = 5

    # Generate all combinations (one per backbone)
    all_combinations = []
    for backbone in AVAILABLE_SMALL_MODELS:
        all_combinations.append({
            "backbone": backbone,
            "n_qubits": n_qubits,
            "template": template,
            "depth": depth
        })

    # Initialize status file with all combinations
    initialize_status_file(all_combinations)

    # Print current status
    print_training_status()

    # Continuous loop to claim and train combinations
    trained_count = 0
    while True:
        # Try to get the next available combination
        model_config, combo_id = get_next_available_combination()

        if model_config is None:
            if trained_count == 0:
                print("No available combinations to train.")
                print("All combinations are either completed or currently being trained by other processes.")
            else:
                print(f"\nThis terminal completed {trained_count} combination(s).")
                print("No more available combinations to claim.")
            print_training_status()
            break

        trained_count += 1

        print("\n" + "=" * 80)
        print(f"CLAIMED COMBINATION: {combo_id}")
        print(f"Model: {base_model}")
        print(f"Backbone: {model_config['backbone']}")
        print(f"Qubits: {model_config['n_qubits']}")
        print(f"Template: {model_config['template']}")
        print(f"Depth: {model_config['depth']}")
        print(f"Encoding: {encoding_method}")
        print("=" * 80)

        # Update config
        config.DEFAULT_MODEL = base_model
        config.QUANTUM_CNN_CONFIG_BACKBONE = model_config['backbone']
        config.QUANTUM_CNN_CONFIG_NO_QUBITS = model_config['n_qubits']
        config.QUANTUM_CNN_CONFIG_DENSE_ENCODING_METHOD = encoding_method
        config.QUANTUM_CNN_CONFIG_DENSE_TEMPLATE = model_config['template']
        config.QUANTUM_CNN_CONFIG_DENSE_DEPTH = model_config['depth']
        config.QUANTUM_CNN_CONFIG_COMBINED_NAME = (
            f"cnn_quantum_"
            f"{model_config['backbone']}_"
            f"dense-{encoding_method}_"
            f"{model_config['template']}_"
            f"depth-{model_config['depth']}_"
            f"qubits-{model_config['n_qubits']}"
        )

        try:
            # Train the model
            main()

            # Clear GPU memory after training
            clear_gpu_memory()

            print(f"\nSuccessfully completed: {combo_id}")

            # Release combination and mark as completed
            release_combination(combo_id, success=True)

        except Exception as e:
            print(f"\nError in {combo_id}: {str(e)}")
            print("Marking as failed and continuing...")

            # Clear GPU memory
            clear_gpu_memory()

            # Release combination and mark as failed
            release_combination(combo_id, success=False)

    print("\n" + "=" * 80)
    print("Training session finished!")
    print("=" * 80)
