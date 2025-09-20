#!/usr/bin/env python
"""
Direct debug of StyleTTS2 training to find where it's exiting.
"""
import os
import sys
import torch

print("=" * 70)
print("Direct StyleTTS2 Training Debug")
print("=" * 70)

# Change to StyleTTS2 directory
os.chdir('StyleTTS2')
sys.path.insert(0, '.')

# Load config
import yaml
config_path = '../data_styletts2/config_somali_ft.yml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print(f"Config loaded from: {config_path}")
print(f"  Device: {config['device']}")
print(f"  Batch size: {config['batch_size']}")
print(f"  Epochs: {config['epochs']}")

# Try to import and run training components step by step
print("\n" + "=" * 70)
print("Testing training components...")

# 1. Test data loading
print("\n1. Testing data loading...")
try:
    from meldataset import build_dataloader

    train_list = config['data_params']['train_data']
    val_list = config['data_params']['val_data']
    root_path = config['data_params']['root_path']

    # Try to build the dataloader
    print(f"   Building train dataloader from {train_list}...")
    train_dataloader = build_dataloader(
        train_list,
        root_path,
        validation=False,
        batch_size=2,  # Small batch for testing
        num_workers=0,  # No multiprocessing for debugging
        device=config['device']
    )

    print(f"   ✓ Train dataloader created")

    # Try to get one batch
    print("   Testing batch loading...")
    for i, batch in enumerate(train_dataloader):
        if i == 0:
            print(f"   ✓ Got batch with {len(batch)} items")
            if isinstance(batch, (list, tuple)):
                for j, item in enumerate(batch):
                    if hasattr(item, 'shape'):
                        print(f"      Item {j}: shape {item.shape}")
            break

except Exception as e:
    print(f"   ✗ Data loading error: {e}")
    import traceback
    traceback.print_exc()

# 2. Test model loading
print("\n2. Testing model loading...")
try:
    from models import *
    from utils import *

    print("   ✓ Imported models and utils")

    # Try to load the models as train_finetune.py does
    print("   Loading pretrained model...")
    pretrained_path = config.get('pretrained_model')
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"   Found pretrained model at: {pretrained_path}")
    else:
        print(f"   ✗ Pretrained model not found at: {pretrained_path}")

except Exception as e:
    print(f"   ✗ Model loading error: {e}")
    import traceback
    traceback.print_exc()

# 3. Check if the training script has any early exit conditions
print("\n3. Checking training script logic...")
try:
    with open('train_finetune.py', 'r') as f:
        train_script = f.read()

    # Look for early exits
    import re

    # Check for sys.exit or return statements in main
    exits = re.findall(r'(sys\.exit|return).*', train_script)
    if exits:
        print(f"   Found {len(exits)} potential exit points")
        for exit_stmt in exits[:5]:  # Show first 5
            print(f"      {exit_stmt.strip()}")

    # Check if there's a condition that might cause early exit
    if '__name__' in train_script and 'if __name__' in train_script:
        print("   ✓ Script has __main__ block")

    # Check for click command
    if '@click.command()' in train_script:
        print("   ✓ Script uses click for CLI")

except Exception as e:
    print(f"   ✗ Error checking script: {e}")

# 4. Try to run the actual training with more verbose output
print("\n4. Attempting actual training initialization...")
print("=" * 70)

try:
    # Import the training script as a module
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_finetune", "train_finetune.py")
    train_module = importlib.util.module_from_spec(spec)

    # Monkey-patch print to ensure we see output
    original_print = print
    def verbose_print(*args, **kwargs):
        original_print("[TRAIN]", *args, **kwargs)
        sys.stdout.flush()

    train_module.print = verbose_print

    # Set command line arguments
    sys.argv = ['train_finetune.py', '--config_path', os.path.abspath(config_path)]

    print("Starting training module execution...")
    print("-" * 70)

    # Execute the module
    spec.loader.exec_module(train_module)

except SystemExit as e:
    print("-" * 70)
    print(f"Training exited with code: {e.code}")
    if e.code == 0:
        print("This appears to be a normal exit (code 0)")
    else:
        print(f"This is an error exit (code {e.code})")
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
except Exception as e:
    print("-" * 70)
    print(f"Training error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Debug complete")

# Check if any output files were created
log_dir = '../Models/Somali'
if os.path.exists(log_dir):
    files = os.listdir(log_dir)
    if files:
        print(f"\nFiles in {log_dir}:")
        for f in files:
            file_path = os.path.join(log_dir, f)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  {f} ({size} bytes)")