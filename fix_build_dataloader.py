#!/usr/bin/env python
"""
Fix the build_dataloader function issue.
"""
import os
import sys

os.chdir('StyleTTS2')
sys.path.insert(0, '.')

print("Investigating build_dataloader issue...")
print("=" * 70)

# Check the build_dataloader function
with open('meldataset.py', 'r') as f:
    lines = f.readlines()

print("\n1. Looking at build_dataloader function...")
for i, line in enumerate(lines):
    if 'def build_dataloader(' in line:
        print(f"Found at line {i+1}")
        # Show the function (next 30 lines)
        for j in range(30):
            if i+j < len(lines):
                print(f"{i+j+1:3}: {lines[i+j].rstrip()}")
        break

print("\n" + "=" * 70)
print("\n2. Testing the issue directly...")

# The problem is that build_dataloader creates a NEW dataset instance
# Let's trace through it
from meldataset import FilePathDataset

train_list_path = '../data_styletts2/train_list.txt'
root_path = '../data_styletts2/wavs'

print(f"Creating dataset with file handle...")
with open(train_list_path, 'r') as f:
    dataset1 = FilePathDataset(
        f,
        root_path,
        sr=24000,
        validation=False,
        OOD_data="../data_styletts2/OOD_list.txt",
        min_length=50
    )
print(f"  Dataset1 length: {len(dataset1)}")

print(f"\nNow what build_dataloader does...")
# build_dataloader opens the file again
print(f"Opening file again (like build_dataloader does)...")
with open(train_list_path, 'r', encoding='utf-8') as f:
    dataset2 = FilePathDataset(
        f,
        root_path,
        sr=24000,
        validation=False,
        OOD_data="../data_styletts2/OOD_list.txt",
        min_length=50
    )
print(f"  Dataset2 length: {len(dataset2)}")

# The issue might be that the dataset filters out samples
# Let's check if there's filtering based on length
print(f"\n3. Checking if samples are filtered...")
print(f"   min_length parameter: 50")

# Let's check audio file durations
import soundfile as sf
sample_count = 0
filtered_count = 0

print(f"\n   Checking first 10 audio files...")
for i in range(min(10, len(dataset1.data_list))):
    audio_file = dataset1.data_list[i][0]
    audio_path = os.path.join(root_path, audio_file)

    if os.path.exists(audio_path):
        try:
            info = sf.info(audio_path)
            duration = info.duration
            num_frames = info.frames
            sample_rate = info.samplerate

            print(f"   {audio_file}: duration={duration:.2f}s, frames={num_frames}, sr={sample_rate}")

            # Check if it meets minimum length requirement
            if num_frames < 50:  # If min_length refers to frames
                print(f"     ⚠ Too short! (< 50 frames)")
                filtered_count += 1
            sample_count += 1
        except Exception as e:
            print(f"   {audio_file}: Error - {e}")
    else:
        print(f"   {audio_file}: File not found!")

if filtered_count > 0:
    print(f"\n   ⚠ {filtered_count}/{sample_count} files might be filtered out!")

print("\n" + "=" * 70)
print("\n4. Checking the actual build_dataloader filtering...")

# Look for any length-based filtering in the dataset
print("\nLooking for filtering code in FilePathDataset...")
for i, line in enumerate(lines):
    if 'FilePathDataset' in line:
        class_start = i
        # Look for __init__ and any filtering
        for j in range(i, min(i+100, len(lines))):
            if 'min_length' in lines[j] and ('if' in lines[j] or '<' in lines[j] or '>' in lines[j]):
                print(f"  Line {j+1}: {lines[j].rstrip()}")
            if 'filter' in lines[j].lower() or 'skip' in lines[j].lower():
                print(f"  Line {j+1}: {lines[j].rstrip()}")

print("\n" + "=" * 70)
print("\n5. Looking at the DataLoader creation in build_dataloader...")

# Find where DataLoader is created
for i, line in enumerate(lines):
    if 'DataLoader(dataset' in line:
        print(f"Found DataLoader creation at line {i+1}")
        # Show context
        for j in range(max(0, i-5), min(i+10, len(lines))):
            print(f"{j+1:3}: {lines[j].rstrip()}")

print("\n" + "=" * 70)
print("\nThe issue might be that build_dataloader creates a dataset but then")
print("something happens to filter it to 0 samples before DataLoader is created.")
print("\nLet me check if there's any sample filtering between dataset creation")
print("and DataLoader instantiation...")