#!/usr/bin/env python
"""
Check what's happening in meldataset.py
"""
import os
import sys

os.chdir('StyleTTS2')
sys.path.insert(0, '.')

print("Checking meldataset.py data loading...")
print("=" * 70)

# Read the train file directly
train_file = '../data_styletts2/train_list.txt'
with open(train_file, 'r') as f:
    lines = f.readlines()

print(f"Raw file has {len(lines)} lines")
print(f"First line raw: {repr(lines[0])}")
print(f"First line stripped: {repr(lines[0].strip())}")
print(f"First line split: {lines[0].strip().split('|')}")

# Now check how FilePathDataset processes it
from meldataset import FilePathDataset

print("\n" + "=" * 70)
print("Testing FilePathDataset initialization...")

# Try to create the dataset
try:
    # Open the file and pass the lines
    with open(train_file, 'r') as f:
        dataset = FilePathDataset(
            f,  # Pass file handle
            '../data_styletts2/wavs',
            sr=24000,
            validation=False,
            OOD_data="../data_styletts2/OOD_list.txt",
            min_length=50
        )

    print(f"Dataset created with {len(dataset)} samples")

    # Check internal data structure
    print(f"\nDataset internal structure:")
    print(f"  data_list type: {type(dataset.data_list)}")
    if len(dataset.data_list) > 0:
        print(f"  First item type: {type(dataset.data_list[0])}")
        print(f"  First item: {dataset.data_list[0]}")
        print(f"  First item length: {len(dataset.data_list[0])}")

    # Try to get one item
    print("\nTrying to get first item...")
    try:
        item = dataset[0]
        print(f"✓ Successfully got item")
        if isinstance(item, tuple):
            print(f"  Item has {len(item)} elements")
    except Exception as e:
        print(f"✗ Error getting item: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"✗ Error creating dataset: {e}")
    import traceback
    traceback.print_exc()

# Check the _load_tensor method specifically
print("\n" + "=" * 70)
print("Checking _load_tensor method...")

# Look at line 139 where the error occurs
with open('meldataset.py', 'r') as f:
    lines = f.readlines()

# Show lines around 139
print("\nmeldataset.py lines 135-145:")
for i in range(135, min(145, len(lines))):
    print(f"{i:3}: {lines[i].rstrip()}")

print("\n" + "=" * 70)
print("\nThe issue is in how data_list items are created.")
print("Line 83 creates tuples, but line 139 expects them to unpack correctly.")