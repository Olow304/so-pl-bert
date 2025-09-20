#!/usr/bin/env python
"""
Fix syntax error and add proper debug output.
"""
import os
import shutil

print("Fixing syntax error and adding proper debug output...")
print("=" * 70)

train_file = 'StyleTTS2/train_finetune.py'

# Restore from backup first
backup_file = train_file + '.debug_backup'
if os.path.exists(backup_file):
    shutil.copy(backup_file, train_file)
    print(f"Restored from backup")

# Read the file
with open(train_file, 'r') as f:
    content = f.read()

# Add debug output more carefully
import re

# 1. After epochs is loaded from config
pattern = r'(epochs = config\.get\([\'"]epochs[\'"][^\)]*\))'
replacement = r'\1\n    print(f"DEBUG: epochs from config = {epochs}")'
content = re.sub(pattern, replacement, content)

# 2. Find where load_checkpoint is called and add debug after it
lines = content.split('\n')
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)

    # After load_checkpoint calls
    if 'load_checkpoint(' in line and '=' in line and not line.strip().startswith('#'):
        # This line calls load_checkpoint and assigns results
        # Add debug on next line with same indentation
        indent = len(line) - len(line.lstrip())

        # Check what variables are assigned
        if 'start_epoch' in line:
            new_lines.append(' ' * indent + 'print(f"DEBUG: After load_checkpoint - start_epoch={start_epoch}")')

content = '\n'.join(new_lines)

# 3. Before the training loop - be more careful
lines = content.split('\n')
new_lines = []
for i, line in enumerate(lines):
    # Before the training loop
    if 'for epoch in range' in line and 'start_epoch' in line:
        indent = len(line) - len(line.lstrip())
        # Add debug before the loop
        new_lines.append(' ' * indent + 'print(f"DEBUG: Before training loop - start_epoch={start_epoch}, epochs={epochs}")')
        new_lines.append(' ' * indent + 'print(f"DEBUG: Range will be range({start_epoch}, {epochs}) = {list(range(start_epoch, epochs))[:5]}...")')

    new_lines.append(line)

    # Inside the training loop
    if 'for epoch in range' in line:
        indent = len(line) - len(line.lstrip()) + 4
        new_lines.append(' ' * indent + 'print(f"DEBUG: Starting epoch {epoch}")')

content = '\n'.join(new_lines)

# Write the modified file
with open(train_file, 'w') as f:
    f.write(content)

print("✓ Fixed and added debug output")

# Also create a simple test to check the range
print("\n" + "=" * 70)
print("Creating a simple test to verify the epoch range...")

test_script = """#!/usr/bin/env python
'''
Test what the epoch range would be
'''
import yaml

config_path = 'data_styletts2/config_somali_ft.yml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

epochs = config.get('epochs', 50)
print(f"epochs from config: {epochs}")

# Simulate what happens with load_checkpoint
# When loading pretrained, start_epoch is typically set to a high value
# if the checkpoint already has many epochs

# Check what the pretrained model might return
pretrained_path = config.get('pretrained_model', '')
print(f"pretrained_model: {pretrained_path}")

# In your case, load_checkpoint might be returning:
# start_epoch = 20 (or some high number from the pretrained model)
# But epochs = 50 from your config

# If start_epoch >= epochs, the loop won't run!
print(f"\\nPossible scenarios:")
for start in [1, 20, 50, 100]:
    print(f"  If start_epoch={start} and epochs={epochs}:")
    print(f"    range({start}, {epochs}) = {list(range(start, epochs))[:5]}")
    if start >= epochs:
        print(f"    ⚠ NO TRAINING WOULD OCCUR!")

print(f"\\nThe issue is likely that start_epoch from the pretrained model")
print(f"is >= {epochs}, causing the training loop to be skipped!")

print(f"\\nSOLUTION: Add 'load_only_params: true' to your config")
print(f"This will load model weights but reset epoch to 1")
"""

with open('test_epoch_range.py', 'w') as f:
    f.write(test_script)

print("\nRun these commands:")
print("  python test_epoch_range.py  # To understand the issue")
print("  python train_gpu.py         # To see debug output")
print("\nThe problem is likely that start_epoch >= epochs!")