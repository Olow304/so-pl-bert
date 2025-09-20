#!/usr/bin/env python
"""
Direct training script that bypasses some complexity.
"""
import os
import sys
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Change to StyleTTS2 directory
os.chdir('StyleTTS2')
sys.path.insert(0, '.')

# Import click first to avoid issues
import click

# Now try to run the training
config_path = os.path.abspath('../data_styletts2/config_somali_ft.yml')

# Set up arguments
sys.argv = ['train_finetune.py', '--config_path', config_path]

print(f"Starting training with config: {config_path}")
print("If this hangs, there may be an issue with data loading...")

# Import and run
try:
    import train_finetune
    # The script should run automatically
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()