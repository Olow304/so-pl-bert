#!/usr/bin/env python
"""
Direct training script for GPU execution.
"""
import os
import sys
import torch

print("=" * 70)
print("StyleTTS2 Fine-tuning with Somali PL-BERT")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: No CUDA available, will use CPU (very slow!)")

# Set environment for better debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

print("\nChecking data files...")
if os.path.exists('data_styletts2/train_list.txt'):
    with open('data_styletts2/train_list.txt', 'r') as f:
        num_samples = len(f.readlines())
    print(f"✓ Found {num_samples} training samples")
else:
    print("✗ Training data not found!")
    sys.exit(1)

if os.path.exists('runs/plbert_so/packaged'):
    print("✓ Somali PL-BERT model found")
else:
    print("✗ Somali PL-BERT not found!")
    sys.exit(1)

print("\nStarting training...")
print("=" * 70)

# Change to StyleTTS2 directory
os.chdir('StyleTTS2')
sys.path.insert(0, '.')

# Import click first to avoid issues
import click

# Now try to run the training
config_path = os.path.abspath('../data_styletts2/config_somali_ft.yml')

# Set up arguments
sys.argv = ['train_finetune.py', '--config_path', config_path]

print(f"Config: {config_path}")
print("\nTraining output:")
print("-" * 70)
sys.stdout.flush()  # Force output

# Import and run
try:
    import train_finetune
    # The script should run automatically
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    print("\nTip: Check Models/Somali/train.log for details")