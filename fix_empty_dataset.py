#!/usr/bin/env python
"""
Fix the empty dataset issue.
"""
import os
import sys

os.chdir('StyleTTS2')
sys.path.insert(0, '.')

print("Debugging empty dataset issue...")
print("=" * 70)

# Check what's happening with the dataset
from meldataset import FilePathDataset

train_list = '../data_styletts2/train_list.txt'
root_path = '../data_styletts2/wavs'

print(f"\n1. Checking raw data file...")
with open(train_list, 'r') as f:
    raw_lines = f.readlines()
print(f"   Raw file has {len(raw_lines)} lines")
print(f"   First line: {raw_lines[0].strip()}")

print(f"\n2. Creating dataset...")
with open(train_list, 'r') as f:
    dataset = FilePathDataset(
        f,
        root_path,
        sr=24000,
        validation=False,
        OOD_data="../data_styletts2/OOD_list.txt",
        min_length=50
    )

print(f"   Dataset created")
print(f"   Dataset length: {len(dataset)}")
print(f"   data_list length: {len(dataset.data_list)}")

if len(dataset.data_list) > 0:
    print(f"   First data_list item: {dataset.data_list[0]}")

# Check what's happening in the initialization
print(f"\n3. Debugging the initialization...")

# Look at meldataset.py to see what might be filtering out data
with open('meldataset.py', 'r') as f:
    lines = f.readlines()

# Find where data_list is modified
print("\nLooking for data filtering code...")
for i, line in enumerate(lines[70:100], 70):
    if 'self.data_list' in line or 'continue' in line or 'skip' in line:
        print(f"  Line {i}: {line.rstrip()}")

print("\n" + "=" * 70)
print("The issue is likely in our fix that uses 'continue' to skip invalid entries")
print("Let me fix it properly...")

# Fix the meldataset.py file
with open('meldataset.py', 'r') as f:
    content = f.read()

# Find and fix the problematic section
old_fix = """        _data_list = [l.strip().split('|') for l in data_list]
        # Ensure all items have exactly 3 elements
        self.data_list = []
        for data in _data_list:
            if len(data) == 2:
                self.data_list.append([data[0], data[1], '0'])
            elif len(data) == 3:
                self.data_list.append(list(data))  # Convert to list
            else:
                continue  # Skip invalid entries"""

# Better fix that doesn't skip valid entries
new_fix = """        _data_list = [l.strip().split('|') for l in data_list]
        # Ensure all items have exactly 3 elements
        self.data_list = []
        for data in _data_list:
            if len(data) >= 2:  # Accept 2 or more elements
                if len(data) == 2:
                    self.data_list.append([data[0], data[1], '0'])
                else:
                    # Take first 3 elements
                    self.data_list.append([data[0], data[1], data[2] if len(data) > 2 else '0'])"""

if old_fix in content:
    content = content.replace(old_fix, new_fix)
    print("\n✓ Fixed data_list initialization to not skip entries")

    # Write the fix
    with open('meldataset.py', 'w') as f:
        f.write(content)
else:
    print("\n⚠ Could not find the exact pattern to fix")
    print("Let me try another approach...")

    # Alternative fix - restore original and modify
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if '_data_list = [l.strip().split' in line:
            # Find the data_list assignment
            for j in range(i, min(i+15, len(lines))):
                if 'self.data_list = []' in lines[j]:
                    # We found our modified section, replace it with working code
                    # Find the end of the for loop
                    indent = len(lines[j]) - len(lines[j].lstrip())

                    # Replace with simpler, working code
                    new_lines = [
                        lines[i],  # Keep the _data_list line
                        ' ' * indent + '# Ensure all items have exactly 3 elements',
                        ' ' * indent + 'self.data_list = []',
                        ' ' * indent + 'for data in _data_list:',
                        ' ' * (indent + 4) + 'if len(data) >= 2:',
                        ' ' * (indent + 8) + 'if len(data) == 2:',
                        ' ' * (indent + 12) + 'self.data_list.append([data[0], data[1], "0"])',
                        ' ' * (indent + 8) + 'else:',
                        ' ' * (indent + 12) + 'self.data_list.append(list(data[:3]))'
                    ]

                    # Find where the for loop ends
                    for k in range(j, min(j+10, len(lines))):
                        if lines[k].strip() and not lines[k].startswith(' ' * (indent + 4)):
                            # Found the line after the for loop
                            # Replace lines from i to k-1
                            lines[i:k] = new_lines
                            content = '\n'.join(lines)

                            with open('meldataset.py', 'w') as f:
                                f.write(content)
                            print("\n✓ Applied alternative fix")
                            break
                    break
            break

# Test again
print("\n" + "=" * 70)
print("Testing the fix...")

# Reimport to get the fixed version
import importlib
import meldataset
importlib.reload(meldataset)

from meldataset import FilePathDataset, build_dataloader

with open(train_list, 'r') as f:
    dataset = FilePathDataset(
        f,
        root_path,
        sr=24000,
        validation=False,
        OOD_data="../data_styletts2/OOD_list.txt",
        min_length=50
    )

print(f"Dataset now has {len(dataset)} samples")

if len(dataset) > 0:
    print("✓ Dataset is no longer empty!")

    # Try to build dataloader
    try:
        train_dataloader = build_dataloader(
            train_list,
            root_path,
            validation=False,
            batch_size=2,
            num_workers=0,
            device='cuda'
        )

        for i, batch in enumerate(train_dataloader):
            print(f"✓ Successfully loaded a batch with {len(batch)} items!")
            break

        print("\n✓✓✓ DATA LOADING FULLY WORKS!")

    except Exception as e:
        print(f"Dataloader error: {e}")
else:
    print("✗ Dataset is still empty - need to investigate further")

print("\nNow run: python train_gpu.py")