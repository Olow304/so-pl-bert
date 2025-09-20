#!/usr/bin/env python
"""
Patch meldataset.py to fix the data loading issue.
"""
import os

print("Patching meldataset.py...")
print("=" * 70)

meldataset_file = 'StyleTTS2/meldataset.py'

# Read the file
with open(meldataset_file, 'r') as f:
    lines = f.readlines()

# Find the problematic line 118-119
print("Looking for the reference sample loading code...")

for i, line in enumerate(lines):
    if 'ref_data = ' in line and 'self.df' in line:
        print(f"Found at line {i+1}: {line.strip()}")

        # Check the next line too
        if i+1 < len(lines):
            print(f"Next line {i+2}: {lines[i+1].strip()}")

print("\nThe issue is that _load_data expects 3 values but ref_data[:3] might only have 2")
print("Let's check the _load_data method...")

# Find _load_data method
for i, line in enumerate(lines):
    if 'def _load_data(' in line:
        print(f"\nFound _load_data at line {i+1}")
        # Show the next few lines
        for j in range(10):
            if i+j < len(lines):
                print(f"{i+j+1:3}: {lines[i+j].rstrip()}")

# Now create the fix
print("\n" + "=" * 70)
print("Applying fix...")

# Backup the original
backup_file = meldataset_file + '.original'
if not os.path.exists(backup_file):
    os.system(f'cp {meldataset_file} {backup_file}')
    print(f"Backed up to {backup_file}")

# The fix: Ensure _load_data gets the right format
# Find and replace the problematic line
modified = False
for i, line in enumerate(lines):
    # Fix line 119 where _load_data is called
    if 'ref_mel_tensor, ref_label = self._load_data(ref_data[:3])' in line:
        # Change to pass ref_data directly since it's already a list of 3 items
        lines[i] = line.replace('self._load_data(ref_data[:3])', 'self._load_data(ref_data)')
        print(f"Fixed line {i+1}: {lines[i].strip()}")
        modified = True

    # Also check if _load_data has issues with unpacking
    if 'def _load_data(self, data):' in line:
        # Look at the next few lines to see how it unpacks
        for j in range(1, 10):
            if i+j < len(lines) and 'wave_path, text' in lines[i+j]:
                # Check if it expects 2 or 3 values
                if 'wave_path, text =' in lines[i+j] and 'speaker_id' not in lines[i+j]:
                    # It expects only 2 values, need to fix this
                    old_line = lines[i+j]
                    lines[i+j] = lines[i+j].replace('wave_path, text =', 'wave_path, text, _ =')
                    print(f"Fixed line {i+j+1}: {lines[i+j].strip()}")
                    print(f"  (was: {old_line.strip()})")
                    modified = True
                    break

if modified:
    # Write the fixed file
    with open(meldataset_file, 'w') as f:
        f.writelines(lines)
    print("\n✓ Patched meldataset.py successfully")
else:
    print("\n⚠ No changes needed or pattern not found")

print("\n" + "=" * 70)
print("Testing if the fix works...")

# Test the dataloader again
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
        break

    print("\n✓ Data loading now works!")

except Exception as e:
    print(f"\n✗ Still getting error: {e}")
    print("\nMight need to check _load_data method more carefully")

print("\nNow try: python train_gpu.py")