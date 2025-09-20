#!/usr/bin/env python
"""
Test if the full data loading pipeline works now.
"""
import os
import sys
import torch

os.chdir('StyleTTS2')
sys.path.insert(0, '.')

print("Testing full data loading pipeline...")
print("=" * 70)

# Test the complete dataloader
from meldataset import build_dataloader

train_list = '../data_styletts2/train_list.txt'
root_path = '../data_styletts2/wavs'

print(f"Building dataloader from:")
print(f"  Train list: {train_list}")
print(f"  Root path: {root_path}")

try:
    # Build the dataloader
    train_dataloader = build_dataloader(
        train_list,
        root_path,
        validation=False,
        batch_size=2,
        num_workers=0,
        device='cuda'
    )

    print(f"\n✓ Dataloader created successfully")

    # Try to get a batch
    print("\nTrying to load a batch...")
    for i, batch in enumerate(train_dataloader):
        print(f"✓ Successfully loaded batch {i}")

        if isinstance(batch, (list, tuple)):
            print(f"  Batch contains {len(batch)} items:")
            for j, item in enumerate(batch):
                if hasattr(item, 'shape'):
                    print(f"    Item {j}: shape {item.shape}, dtype {item.dtype}")
                else:
                    print(f"    Item {j}: type {type(item)}")

        # Just test one batch
        break

    print("\n✓ Data loading works!")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

    # Try to understand what's failing
    print("\nDebugging the error...")

    # Check if it's still the unpacking issue
    if "unpack" in str(e):
        print("\nThe unpacking error persists. Let's check line 139 of meldataset.py")

        # Create a minimal test
        from meldataset import FilePathDataset

        with open(train_list, 'r') as f:
            dataset = FilePathDataset(
                f,
                root_path,
                sr=24000,
                validation=False,
                OOD_data="../data_styletts2/OOD_list.txt",
                min_length=50
            )

        # Get the problematic data item
        print(f"\nDataset has {len(dataset)} items")
        print(f"First data item: {dataset.data_list[0]}")
        print(f"Type: {type(dataset.data_list[0])}")

        # Try calling _load_tensor directly
        print("\nTrying _load_tensor directly...")
        try:
            result = dataset._load_tensor(dataset.data_list[0])
            print(f"✓ _load_tensor succeeded")
        except Exception as e2:
            print(f"✗ _load_tensor failed: {e2}")

            # The problem might be that data_list[0] is getting modified somewhere
            print(f"\nChecking if data gets modified...")
            print(f"dataset.data_list[0] now: {dataset.data_list[0]}")

print("\n" + "=" * 70)
print("Test complete")