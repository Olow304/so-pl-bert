#!/usr/bin/env python
"""
Investigate why training exits after initialization.
"""
import os
import sys

print("Investigating training exit issue...")
print("=" * 70)

# Look at the train_finetune.py to see what happens after optimizer creation
train_file = 'StyleTTS2/train_finetune.py'

with open(train_file, 'r') as f:
    lines = f.readlines()

print("\n1. Looking for what happens after optimizer initialization...")

# Find where optimizers are created
for i, line in enumerate(lines):
    if 'decoder AdamW' in line or 'BERT AdamW' in line or 'print("decoder AdamW' in line:
        print(f"Found optimizer print at line {i+1}")

        # Show the next 20 lines to see what happens next
        print("\nNext 20 lines after optimizer initialization:")
        for j in range(i+1, min(i+21, len(lines))):
            print(f"{j+1:4}: {lines[j].rstrip()}")
        break

print("\n" + "=" * 70)
print("\n2. Looking for training loop...")

# Find the training loop
for i, line in enumerate(lines):
    if 'for epoch in' in line:
        print(f"Found training loop at line {i+1}: {line.strip()}")

        # Check what's before the loop that might exit
        print("\n10 lines before the training loop:")
        for j in range(max(0, i-10), i):
            stripped = lines[j].strip()
            if stripped and not stripped.startswith('#'):
                print(f"{j+1:4}: {lines[j].rstrip()}")

        # Check the loop condition
        print(f"\nLoop line: {lines[i].rstrip()}")
        break

print("\n" + "=" * 70)
print("\n3. Looking for early return conditions...")

# Look for conditions that might cause early exit
for i, line in enumerate(lines):
    if 'return' in line and not line.strip().startswith('#'):
        # Check if it's inside the main function
        # Get context
        context_start = max(0, i-3)
        context_end = min(len(lines), i+2)

        print(f"\nFound return at line {i+1}:")
        for j in range(context_start, context_end):
            marker = ">>>" if j == i else "   "
            print(f"{marker} {j+1:4}: {lines[j].rstrip()}")

print("\n" + "=" * 70)
print("\n4. Checking if epochs is set correctly...")

config_file = 'data_styletts2/config_somali_ft.yml'
if os.path.exists(config_file):
    import yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Config settings:")
    print(f"  epochs: {config.get('epochs', 'NOT SET')}")
    print(f"  batch_size: {config.get('batch_size', 'NOT SET')}")

    # Check if epochs might be 0 or negative
    if config.get('epochs', 0) <= 0:
        print("  ⚠ EPOCHS IS 0 OR NEGATIVE - this would cause immediate exit!")

print("\n" + "=" * 70)
print("\n5. Creating a patched training script...")

# Create a simple patch that adds debug output
patch_script = """#!/usr/bin/env python
'''
Patch train_finetune.py to add debug output
'''
import os

print("Patching train_finetune.py to add debug output...")

train_file = 'StyleTTS2/train_finetune.py'

with open(train_file, 'r') as f:
    content = f.read()

# Add debug prints
patches = [
    # After config loading
    ('config = yaml.safe_load(f)',
     'config = yaml.safe_load(f)\\nprint(f"DEBUG: Loaded config, epochs={config.get(\\'epochs\\', \\'NOT SET\\')}")'),

    # Before training loop
    ('for epoch in range(1, epochs + 1):',
     'print(f"DEBUG: About to start training loop, epochs={epochs}")\\n    for epoch in range(1, epochs + 1):\\n        print(f"DEBUG: Starting epoch {epoch}/{epochs}")'),

    # If there's a different loop format
    ('for epoch in range(epochs):',
     'print(f"DEBUG: About to start training loop, epochs={epochs}")\\n    for epoch in range(epochs):\\n        print(f"DEBUG: Starting epoch {epoch}/{epochs}")')
]

modified = False
for old, new in patches:
    if old in content and new not in content:
        content = content.replace(old, new)
        modified = True
        print(f"  Added debug output for: {old[:30]}...")

if modified:
    # Backup original
    import shutil
    shutil.copy(train_file, train_file + '.bak')

    # Write patched version
    with open(train_file, 'w') as f:
        f.write(content)

    print("✓ Patched train_finetune.py with debug output")
else:
    print("⚠ Could not patch - patterns not found or already patched")

print("\\nNow run: python train_gpu.py")
print("You should see DEBUG messages showing what's happening")
"""

with open('patch_training.py', 'w') as f:
    f.write(patch_script)

print("Created patch_training.py")
print("\nRun these commands:")
print("  python patch_training.py")
print("  python train_gpu.py")
print("\nThis will add debug output to help us see why training exits")