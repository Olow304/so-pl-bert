#!/usr/bin/env python
"""
Final fix for the dataloader issue.
"""
import os

print("Applying final fix to meldataset.py...")
print("=" * 70)

meldataset_file = 'StyleTTS2/meldataset.py'

# Read the file
with open(meldataset_file, 'r') as f:
    lines = f.readlines()

print("Finding and fixing build_dataloader function...")

# Find the build_dataloader function
for i, line in enumerate(lines):
    if 'def build_dataloader(' in line:
        print(f"Found build_dataloader at line {i+1}")

        # Find the dataset creation line
        for j in range(i, min(i+20, len(lines))):
            if 'dataset = FilePathDataset(' in lines[j]:
                print(f"Found dataset creation at line {j+1}")
                print(f"Current: {lines[j].strip()}")

                # The issue is that path_list is a string path, but FilePathDataset expects a file handle
                # We need to open the file first
                indent = len(lines[j]) - len(lines[j].lstrip())

                # Insert file opening before dataset creation
                new_lines = [
                    ' ' * indent + '# Open the file if path_list is a string path\n',
                    ' ' * indent + 'if isinstance(path_list, str):\n',
                    ' ' * (indent + 4) + 'with open(path_list, "r", encoding="utf-8") as f:\n',
                    ' ' * (indent + 8) + 'dataset = FilePathDataset(f, root_path, OOD_data=OOD_data, min_length=min_length, validation=validation, **dataset_config)\n',
                    ' ' * indent + 'else:\n',
                    ' ' * (indent + 4) + 'dataset = FilePathDataset(path_list, root_path, OOD_data=OOD_data, min_length=min_length, validation=validation, **dataset_config)\n'
                ]

                # Replace the single line with our new lines
                lines[j] = ''.join(new_lines)

                print(f"Fixed to handle both file paths and file handles")
                break
        break

# Write the fixed file
with open(meldataset_file, 'w') as f:
    f.writelines(lines)

print("\n✓ Fixed build_dataloader to properly handle file paths")

# Test the fix
print("\n" + "=" * 70)
print("Testing the complete fix...")

os.chdir('StyleTTS2')
import sys
sys.path.insert(0, '.')

try:
    from meldataset import build_dataloader

    print("\n1. Testing with training data...")
    train_dataloader = build_dataloader(
        '../data_styletts2/train_list.txt',
        '../data_styletts2/wavs',
        validation=False,
        batch_size=2,
        num_workers=0,
        device='cuda'
    )

    print(f"   ✓ Train dataloader created")

    # Try to get a batch
    for i, batch in enumerate(train_dataloader):
        print(f"   ✓ Successfully loaded batch with {len(batch)} items!")
        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            for j, item in enumerate(batch[:3]):
                if hasattr(item, 'shape'):
                    print(f"      Item {j}: shape {item.shape}")
        break

    print("\n2. Testing with validation data...")
    val_dataloader = build_dataloader(
        '../data_styletts2/val_list.txt',
        '../data_styletts2/wavs',
        validation=True,
        batch_size=2,
        num_workers=0,
        device='cuda'
    )

    print(f"   ✓ Val dataloader created")

    for i, batch in enumerate(val_dataloader):
        print(f"   ✓ Successfully loaded val batch!")
        break

    print("\n" + "=" * 70)
    print("✓✓✓ SUCCESS! Data loading is completely fixed!")
    print("=" * 70)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

    print("\nDebugging further...")

    # Try to understand what's happening
    train_list = '../data_styletts2/train_list.txt'

    # Direct test
    from meldataset import FilePathDataset

    print(f"\nDirect test with file handle:")
    with open(train_list, 'r') as f:
        dataset = FilePathDataset(
            f,
            '../data_styletts2/wavs',
            sr=24000,
            validation=False,
            OOD_data="../data_styletts2/OOD_list.txt",
            min_length=50
        )
    print(f"Dataset length: {len(dataset)}")

print("\n" + "=" * 70)
print("Now run: python train_gpu.py")
print("Training should finally work!")