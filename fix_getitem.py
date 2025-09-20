#!/usr/bin/env python
"""
Debug and fix the __getitem__ method issue.
"""
import os
import sys

os.chdir('StyleTTS2')
sys.path.insert(0, '.')

print("Debugging __getitem__ method...")
print("=" * 70)

# Look at the __getitem__ method
with open('meldataset.py', 'r') as f:
    lines = f.readlines()

print("Lines 100-120 of meldataset.py (__getitem__ method):")
for i in range(100, min(120, len(lines))):
    print(f"{i:3}: {lines[i].rstrip()}")

print("\n" + "=" * 70)
print("\nTesting the issue:")

from meldataset import FilePathDataset

train_list = '../data_styletts2/train_list.txt'
root_path = '../data_styletts2/wavs'

with open(train_list, 'r') as f:
    dataset = FilePathDataset(
        f,
        root_path,
        sr=24000,
        validation=False,
        OOD_data="../data_styletts2/OOD_list.txt",
        min_length=50
    )

print(f"Dataset created with {len(dataset)} items")

# Check the DataFrame
print(f"\nDataset DataFrame info:")
print(f"  Shape: {dataset.df.shape}")
print(f"  Columns: {dataset.df.columns.tolist()}")
print(f"  First row:")
print(f"    {dataset.df.iloc[0].tolist()}")

# The issue might be in how the DataFrame is created or accessed
idx = 0
print(f"\nTrying to get item at index {idx}:")

# Manually do what __getitem__ does
data = dataset.data_list[idx]
print(f"  data_list[{idx}]: {data}")

# Check if it's the df.iloc that's causing issues
df_data = dataset.df.iloc[idx]
print(f"  df.iloc[{idx}]: {df_data.tolist()}")
print(f"  Type: {type(df_data)}")
print(f"  Length when converted to list: {len(df_data.tolist())}")

# The problem is likely that df.iloc returns a Series, and when converted
# it might not have the right format

print("\n" + "=" * 70)
print("Solution: The issue is likely in line 109-110 where it uses df.iloc")
print("instead of data_list. Let me create a patch...")