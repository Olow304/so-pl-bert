#!/usr/bin/env python
"""
Comprehensive fix for meldataset.py
"""
import os

print("Applying comprehensive fix to meldataset.py...")
print("=" * 70)

meldataset_file = 'StyleTTS2/meldataset.py'

# Read the current file
with open(meldataset_file, 'r') as f:
    content = f.read()

# Backup if not already done
backup_file = meldataset_file + '.backup_comprehensive'
if not os.path.exists(backup_file):
    with open(backup_file, 'w') as f:
        f.write(content)
    print(f"Backed up to {backup_file}")

# Apply multiple fixes
print("\nApplying fixes...")

# Fix 1: Make _load_tensor more robust
old_load_tensor = """    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)"""

new_load_tensor = """    def _load_tensor(self, data):
        # Handle both 2 and 3 element data
        if len(data) == 2:
            wave_path, text = data
            speaker_id = 0  # Default speaker
        elif len(data) == 3:
            wave_path, text, speaker_id = data
            speaker_id = int(speaker_id)
        else:
            raise ValueError(f"Expected 2 or 3 elements in data, got {len(data)}")"""

if old_load_tensor in content:
    content = content.replace(old_load_tensor, new_load_tensor)
    print("✓ Fixed _load_tensor to handle both 2 and 3 element data")
else:
    print("⚠ Could not find exact _load_tensor pattern, trying alternative fix...")

    # Alternative: Find and replace the line more carefully
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'def _load_tensor(self, data):' in line:
            # Find the unpacking line
            for j in range(i+1, min(i+10, len(lines))):
                if 'wave_path, text, speaker_id = data' in lines[j]:
                    # Replace with safer unpacking
                    indent = len(lines[j]) - len(lines[j].lstrip())
                    new_lines = [
                        ' ' * indent + '# Handle both 2 and 3 element data',
                        ' ' * indent + 'if len(data) == 2:',
                        ' ' * indent + '    wave_path, text = data',
                        ' ' * indent + '    speaker_id = 0  # Default speaker',
                        ' ' * indent + 'elif len(data) == 3:',
                        ' ' * indent + '    wave_path, text, speaker_id = data',
                        ' ' * indent + '    speaker_id = int(speaker_id)',
                        ' ' * indent + 'else:',
                        ' ' * indent + '    raise ValueError(f"Expected 2 or 3 elements in data, got {len(data)}")',
                    ]

                    # Replace the line and the next line (speaker_id = int(speaker_id))
                    lines[j] = '\n'.join(new_lines)
                    if j+1 < len(lines) and 'speaker_id = int(speaker_id)' in lines[j+1]:
                        lines[j+1] = ''  # Remove the redundant line

                    content = '\n'.join(lines)
                    print("✓ Fixed _load_tensor with alternative method")
                    break
            break

# Fix 2: Make sure data_list is consistently formatted
# Find the line where data_list is created (around line 82-83)
old_data_list = """        _data_list = [l.strip().split('|') for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]"""

new_data_list = """        _data_list = [l.strip().split('|') for l in data_list]
        # Ensure all items have exactly 3 elements
        self.data_list = []
        for data in _data_list:
            if len(data) == 2:
                self.data_list.append([data[0], data[1], '0'])
            elif len(data) == 3:
                self.data_list.append(list(data))  # Convert to list
            else:
                continue  # Skip invalid entries"""

if old_data_list in content:
    content = content.replace(old_data_list, new_data_list)
    print("✓ Fixed data_list initialization")
else:
    print("⚠ Could not find exact data_list pattern")

# Write the fixed content
with open(meldataset_file, 'w') as f:
    f.write(content)

print("\n✓ Applied comprehensive fixes to meldataset.py")

# Test the fix
print("\n" + "=" * 70)
print("Testing the fix...")

os.chdir('StyleTTS2')
import sys
sys.path.insert(0, '.')

try:
    from meldataset import build_dataloader

    train_dataloader = build_dataloader(
        '../data_styletts2/train_list.txt',
        '../data_styletts2/wavs',
        validation=False,
        batch_size=2,
        num_workers=0,
        device='cuda'
    )

    # Try to get a batch
    for i, batch in enumerate(train_dataloader):
        print(f"✓ Successfully loaded batch!")
        if isinstance(batch, (list, tuple)):
            print(f"  Batch has {len(batch)} items")
            for j, item in enumerate(batch[:3]):  # Show first 3 items
                if hasattr(item, 'shape'):
                    print(f"    Item {j}: shape {item.shape}")
        break

    print("\n✓✓✓ Data loading WORKS! Training should now run!")

except Exception as e:
    print(f"\n✗ Still getting error: {e}")
    print("\nLet me know the exact error and I'll fix it")

print("\n" + "=" * 70)
print("Now run: python train_gpu.py")