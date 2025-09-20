#!/usr/bin/env python
"""
Fix speaker IDs to be numeric instead of "somali".
"""
import os

print("Fixing speaker IDs to be numeric...")
print("=" * 70)

files_to_fix = [
    'data_styletts2/train_list.txt',
    'data_styletts2/val_list.txt',
    'data_styletts2/OOD_list.txt'
]

for file_path in files_to_fix:
    print(f"\nProcessing {file_path}...")

    if not os.path.exists(file_path):
        print(f"  ✗ File not found")
        continue

    with open(file_path, 'r') as f:
        lines = f.readlines()

    print(f"  Found {len(lines)} lines")

    # Fix each line
    fixed_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split('|')
        if len(parts) == 3:
            # Replace "somali" with "0" (single speaker)
            if parts[2] == 'somali':
                parts[2] = '0'
            elif not parts[2].isdigit():
                # If it's any other non-numeric value, use 0
                parts[2] = '0'

            fixed_line = '|'.join(parts) + '\n'
            fixed_lines.append(fixed_line)
        else:
            print(f"  ⚠ Unexpected format: {line[:100]}...")

    # Backup original
    backup_path = file_path + '.with_somali'
    if not os.path.exists(backup_path):
        os.rename(file_path, backup_path)
        print(f"  Backed up to: {backup_path}")

    # Write fixed file
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)

    print(f"  ✓ Fixed {len(fixed_lines)} lines")

    # Show sample
    print("  Sample fixed lines:")
    for i, line in enumerate(fixed_lines[:3]):
        parts = line.strip().split('|')
        print(f"    {parts[0][:20]}... | {parts[1][:30]}... | {parts[2]}")

print("\n" + "=" * 70)
print("Speaker IDs fixed! Now all are numeric (0 for single speaker)")
print("Run: python train_gpu.py")