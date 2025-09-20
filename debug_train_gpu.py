#!/usr/bin/env python
"""
Debug training script to find why it exits immediately.
"""
import os
import sys
import torch
import traceback

print("=" * 70)
print("DEBUG: StyleTTS2 Training Diagnostic")
print("=" * 70)

# Change to StyleTTS2 directory first
if os.path.exists('StyleTTS2'):
    os.chdir('StyleTTS2')
    sys.path.insert(0, '.')
    print("✓ Changed to StyleTTS2 directory")
else:
    print("✗ StyleTTS2 directory not found!")
    sys.exit(1)

# Try importing the training script step by step
print("\nStep 1: Importing base modules...")
try:
    import yaml
    import numpy as np
    print("✓ Base imports OK")
except Exception as e:
    print(f"✗ Base import error: {e}")
    sys.exit(1)

print("\nStep 2: Loading config...")
config_path = '../data_styletts2/config_somali_ft.yml'
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded: {config_path}")
    print(f"  Device: {config.get('device', 'not set')}")
    print(f"  Batch size: {config.get('batch_size', 'not set')}")
except Exception as e:
    print(f"✗ Config error: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nStep 3: Testing data loader...")
try:
    from meldataset import build_dataloader
    print("✓ Can import meldataset")

    # Try to actually build the dataloader
    print("  Building train dataloader...")
    train_list = config['data_params']['train_data']
    root_path = config['data_params']['root_path']

    # Check if files exist
    if not os.path.exists(train_list):
        train_list = '../' + train_list
    if not os.path.exists(root_path):
        root_path = '../' + root_path

    print(f"  Train list: {train_list}")
    print(f"  Root path: {root_path}")

    # Try minimal dataloader
    from meldataset import Dataset
    dataset = Dataset(
        train_list,
        root_path,
        24000,  # sample rate
        validation=False
    )
    print(f"✓ Dataset created with {len(dataset)} samples")

except Exception as e:
    print(f"✗ Data loader error: {e}")
    traceback.print_exc()

print("\nStep 4: Testing model imports...")
try:
    import models
    import utils
    print("✓ Can import models and utils")
except Exception as e:
    print(f"✗ Model import error: {e}")
    traceback.print_exc()

print("\nStep 5: Checking train_finetune.py...")
try:
    # Check if the file exists
    if os.path.exists('train_finetune.py'):
        print("✓ train_finetune.py exists")

        # Try to import it
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_finetune", "train_finetune.py")
        train_module = importlib.util.module_from_spec(spec)

        print("  Checking if it's a runnable script...")
        with open('train_finetune.py', 'r') as f:
            content = f.read()
            if '__main__' in content:
                print("✓ Has __main__ block")

                # Check for click decorator
                if '@click.command()' in content:
                    print("✓ Uses click for arguments")

                    # Now actually try to run it
                    print("\nStep 6: Attempting to run training...")
                    sys.argv = ['train_finetune.py', '--config_path', os.path.abspath(config_path)]

                    try:
                        spec.loader.exec_module(train_module)
                    except SystemExit as e:
                        print(f"SystemExit with code: {e.code}")
                        if e.code == 0:
                            print("Training completed successfully")
                        else:
                            print("Training exited with error")
                    except Exception as e:
                        print(f"✗ Training error: {e}")
                        traceback.print_exc()
                else:
                    print("✗ Doesn't use click")
            else:
                print("✗ No __main__ block")
    else:
        print("✗ train_finetune.py not found!")
except Exception as e:
    print(f"✗ Error checking train_finetune: {e}")
    traceback.print_exc()

print("\n" + "=" * 70)
print("Diagnostic complete")