#!/usr/bin/env python
"""
Debug script to check if training is actually running.
"""
import os
import subprocess
import time

def check_training_status():
    """Check if training is running and what's happening."""

    # Check for log files
    log_dir = "Models/Somali"
    if os.path.exists(log_dir):
        print(f"Log directory exists: {log_dir}")
        files = os.listdir(log_dir)
        if files:
            print(f"Files in log dir: {files}")
        else:
            print("No files in log directory yet")
    else:
        print(f"Log directory not created yet: {log_dir}")

    # Check for tensorboard logs
    tb_dir = "Models/Somali/logs"
    if os.path.exists(tb_dir):
        print(f"Tensorboard logs exist: {tb_dir}")

    # Check GPU usage
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if 'python' in result.stdout.lower():
            print("\n✓ Python process is using GPU")
        else:
            print("\n⚠ No Python process on GPU")
    except:
        print("Could not check GPU status")

    # Check if data files exist
    train_file = "data_styletts2/train_list.txt"
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            lines = f.readlines()
        print(f"\n✓ Training data found: {len(lines)} samples")

        # Check if audio files exist
        first_line = lines[0].strip()
        audio_file = first_line.split('|')[0]
        audio_path = os.path.join("data_styletts2/wavs", audio_file)
        if os.path.exists(audio_path):
            print(f"✓ Audio files accessible: {audio_path}")
        else:
            print(f"⚠ Audio file not found: {audio_path}")

    # Check for error logs
    if os.path.exists("error.log"):
        print("\n⚠ Error log found!")
        with open("error.log", 'r') as f:
            print(f.read()[-500:])  # Last 500 chars

if __name__ == "__main__":
    check_training_status()