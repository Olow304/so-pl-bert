#!/usr/bin/env python
"""
Check the current format of data files.
"""
import os

print("Checking data file formats...")
print("=" * 70)

files_to_check = [
    'data_styletts2/train_list.txt',
    'data_styletts2/val_list.txt',
    'data_styletts2/OOD_list.txt'
]

for file_path in files_to_check:
    print(f"\n{file_path}:")

    if not os.path.exists(file_path):
        print("  ✗ File not found")
        continue

    with open(file_path, 'r') as f:
        lines = f.readlines()

    print(f"  Total lines: {len(lines)}")

    # Check first 5 lines
    print("  First 5 lines:")
    for i, line in enumerate(lines[:5]):
        line = line.strip()
        if not line:
            print(f"    Line {i}: [empty]")
            continue

        parts = line.split('|')
        print(f"    Line {i}: {len(parts)} parts")

        if len(parts) >= 1:
            print(f"      [0] Audio: {parts[0][:40]}...")
        if len(parts) >= 2:
            print(f"      [1] Text: {parts[1][:40]}...")
        if len(parts) >= 3:
            print(f"      [2] Speaker: {parts[2]}")

        if len(parts) < 3:
            print(f"      ⚠ MISSING SPEAKER ID - only {len(parts)} parts!")

print("\n" + "=" * 70)
print("\nTo fix the format, run:")
print("  python fix_data_format.py")