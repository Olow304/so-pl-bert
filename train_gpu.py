#!/usr/bin/env python
"""
GPU training script that properly launches StyleTTS2 fine-tuning.
"""
import os
import sys
import torch
import subprocess

print("=" * 70)
print("StyleTTS2 Fine-tuning with Somali PL-BERT")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Set environment for better GPU performance
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

print("\nChecking prerequisites...")

# Check data
if os.path.exists('data_styletts2/train_list.txt'):
    with open('data_styletts2/train_list.txt', 'r') as f:
        num_samples = len(f.readlines())
    print(f"✓ Found {num_samples} training samples")
else:
    print("✗ Training data not found!")
    sys.exit(1)

# Check PL-BERT
if os.path.exists('runs/plbert_so/packaged'):
    print("✓ Somali PL-BERT model found")
else:
    print("✗ Somali PL-BERT not found!")
    sys.exit(1)

# Check pretrained model
if os.path.exists('Models/LibriTTS/epochs_2nd_00020_fixed.pth'):
    print("✓ Pretrained StyleTTS2 model found")
else:
    print("✗ Pretrained model not found!")
    print("  Please ensure Models/LibriTTS/epochs_2nd_00020_fixed.pth exists")

print("\n" + "=" * 70)
print("Starting training process...")
print("=" * 70)

# Use subprocess to run training with proper output
config_path = os.path.abspath('data_styletts2/config_somali_ft.yml')

# Change to StyleTTS2 directory and run
os.chdir('StyleTTS2')

cmd = [
    sys.executable,  # Use same Python interpreter
    "train_finetune.py",
    "--config_path", config_path
]

print(f"Command: {' '.join(cmd)}")
print("\nTraining output:")
print("-" * 70)
sys.stdout.flush()

try:
    # Run with real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    # Print output line by line
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        sys.stdout.flush()

    # Wait for completion
    process.wait()

    if process.returncode == 0:
        print("\n" + "=" * 70)
        print("Training completed successfully!")
    else:
        print("\n" + "=" * 70)
        print(f"Training exited with code: {process.returncode}")
        print("Check Models/Somali/train.log for details")

except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")
    process.terminate()
    process.wait()
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()