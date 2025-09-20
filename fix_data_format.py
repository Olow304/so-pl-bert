#!/usr/bin/env python
"""
Fix the data format to include speaker IDs for StyleTTS2.
"""
import os

print("Fixing data format for StyleTTS2...")
print("=" * 70)

# Check current format
train_file = 'data_styletts2/train_list.txt'
val_file = 'data_styletts2/val_list.txt'

def fix_data_file(file_path):
    """Fix a data file to have the correct format."""
    print(f"\nProcessing {file_path}...")

    if not os.path.exists(file_path):
        print(f"  ✗ File not found: {file_path}")
        return

    with open(file_path, 'r') as f:
        lines = f.readlines()

    print(f"  Found {len(lines)} lines")

    # Check the format of the first few lines
    print("  Sample lines (original):")
    for i, line in enumerate(lines[:3]):
        parts = line.strip().split('|')
        print(f"    Line {i}: {len(parts)} parts")
        if len(parts) >= 2:
            print(f"      Audio: {parts[0][:50]}...")
            print(f"      Text: {parts[1][:50]}...")
            if len(parts) >= 3:
                print(f"      Speaker: {parts[2]}")

    # Fix lines that don't have speaker ID
    fixed_lines = []
    needs_fixing = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split('|')

        if len(parts) == 2:
            # Add default speaker ID (0 for single speaker)
            fixed_line = f"{parts[0]}|{parts[1]}|0\n"
            fixed_lines.append(fixed_line)
            needs_fixing = True
        elif len(parts) == 3:
            # Already has speaker ID
            fixed_lines.append(line + '\n')
        else:
            print(f"  ⚠ Unexpected format: {line[:100]}...")
            # Try to handle it anyway
            if len(parts) > 3:
                # Take first 3 parts
                fixed_line = f"{parts[0]}|{parts[1]}|{parts[2]}\n"
                fixed_lines.append(fixed_line)
                needs_fixing = True

    if needs_fixing:
        # Backup original file
        backup_path = file_path + '.original'
        if not os.path.exists(backup_path):
            os.rename(file_path, backup_path)
            print(f"  Backed up original to: {backup_path}")

        # Write fixed file
        with open(file_path, 'w') as f:
            f.writelines(fixed_lines)

        print(f"  ✓ Fixed {len(fixed_lines)} lines")

        # Show sample of fixed format
        print("  Sample lines (fixed):")
        for i, line in enumerate(fixed_lines[:3]):
            parts = line.strip().split('|')
            print(f"    Line {i}: {len(parts)} parts - {parts[0][:30]}... | {parts[1][:30]}... | {parts[2]}")
    else:
        print("  ✓ Format already correct")

# Fix both files
fix_data_file(train_file)
fix_data_file(val_file)

# Also check OOD file
ood_file = 'data_styletts2/OOD_list.txt'
if os.path.exists(ood_file):
    fix_data_file(ood_file)
else:
    print(f"\n⚠ OOD file not found: {ood_file}")
    print("  Creating minimal OOD file...")

    # Create a minimal OOD file with a few samples from validation
    if os.path.exists(val_file):
        with open(val_file, 'r') as f:
            val_lines = f.readlines()

        # Take first 10 lines for OOD
        ood_lines = val_lines[:10] if len(val_lines) >= 10 else val_lines

        with open(ood_file, 'w') as f:
            f.writelines(ood_lines)

        print(f"  ✓ Created OOD file with {len(ood_lines)} samples")

print("\n" + "=" * 70)
print("Data format fixed! Now ready for training.")
print("Run: python train_gpu.py")