#!/usr/bin/env python
"""
Deep debug of the data loading issue.
"""
import os
import sys

print("Deep debugging data loading issue...")
print("=" * 70)

# First check the actual file content
train_file = 'data_styletts2/train_list.txt'
print(f"\n1. Checking {train_file} directly:")
with open(train_file, 'r') as f:
    first_5_lines = [f.readline() for _ in range(5)]

for i, line in enumerate(first_5_lines):
    parts = line.strip().split('|')
    print(f"  Line {i}: {len(parts)} parts -> {parts}")

# Now check what meldataset.py sees
os.chdir('StyleTTS2')
sys.path.insert(0, '.')

print("\n2. Checking inside meldataset.py logic:")

# Read the meldataset.py file to understand the issue
with open('meldataset.py', 'r') as f:
    lines = f.readlines()

# Look at how data is loaded (around line 82-83)
print("\n  Lines 80-85 of meldataset.py (data loading):")
for i in range(80, 86):
    if i < len(lines):
        print(f"  {i}: {lines[i].rstrip()}")

print("\n3. Testing the actual data loading code:")

# Simulate what meldataset does
data_list_path = '../data_styletts2/train_list.txt'
with open(data_list_path, 'r') as f:
    data_list = f.readlines()  # This is what gets passed to FilePathDataset

print(f"  Read {len(data_list)} lines from file")

# Simulate line 82 of meldataset.py
_data_list = [l.strip().split('|') for l in data_list]
print(f"  After strip().split('|'): {len(_data_list)} items")
print(f"  First item: {_data_list[0]}")
print(f"  First item length: {len(_data_list[0])}")

# Simulate line 83 - this is where the problem might be
processed_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
print(f"  After processing: {len(processed_list)} items")
print(f"  First processed item: {processed_list[0]}")
print(f"  Type of first item: {type(processed_list[0])}")

# Now test the actual FilePathDataset
print("\n4. Testing FilePathDataset class:")
try:
    from meldataset import FilePathDataset

    # FilePathDataset expects a file object, not a path
    with open(data_list_path, 'r') as f:
        dataset = FilePathDataset(
            f,  # File object
            '../data_styletts2/wavs',
            sr=24000,
            validation=False,
            OOD_data="../data_styletts2/OOD_list.txt",
            min_length=50
        )

    print(f"  Dataset created with {len(dataset)} items")

    # Check the internal data structure
    if hasattr(dataset, 'data_list'):
        print(f"  dataset.data_list length: {len(dataset.data_list)}")
        print(f"  dataset.data_list[0]: {dataset.data_list[0]}")
        print(f"  Type: {type(dataset.data_list[0])}")

        # Check if it's a tuple that needs to be converted to list
        if isinstance(dataset.data_list[0], tuple):
            print("  ⚠ Data is stored as tuples, might need conversion to list")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Analysis complete. The issue is likely in how data_list items are")
print("stored (as tuples) vs how they're accessed (expecting lists).")