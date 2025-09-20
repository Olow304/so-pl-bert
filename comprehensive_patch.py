#!/usr/bin/env python
"""
Comprehensive patch to debug why training exits.
"""
import os
import shutil

print("Applying comprehensive debug patch to train_finetune.py...")
print("=" * 70)

train_file = 'StyleTTS2/train_finetune.py'

# Backup original
backup_file = train_file + '.debug_backup'
if not os.path.exists(backup_file):
    shutil.copy(train_file, backup_file)
    print(f"Backed up to {backup_file}")

with open(train_file, 'r') as f:
    lines = f.readlines()

# Find key locations and add debug output
modified = False

for i, line in enumerate(lines):
    # Add debug after epochs is set
    if 'epochs = config.get' in line or 'epochs = config[' in line:
        if i+1 < len(lines) and 'DEBUG:' not in lines[i+1]:
            lines.insert(i+1, f"    print(f'DEBUG: epochs from config = {{epochs}}')\n")
            modified = True
            print(f"Added debug after epochs assignment at line {i+1}")

    # Add debug after start_epoch is set
    if 'start_epoch' in line and '=' in line and 'DEBUG:' not in line:
        if i+1 < len(lines) and 'DEBUG:' not in lines[i+1]:
            indent = len(line) - len(line.lstrip())
            lines.insert(i+1, ' ' * indent + f"print(f'DEBUG: start_epoch = {{start_epoch}}')\n")
            modified = True
            print(f"Added debug after start_epoch assignment at line {i+1}")

    # Add debug before the training loop
    if 'for epoch in range' in line and 'DEBUG:' not in lines[i-1]:
        indent = len(line) - len(line.lstrip())
        debug_line = ' ' * indent + "print(f'DEBUG: About to start training loop - start_epoch={start_epoch}, epochs={epochs}, range({start_epoch}, {epochs})')\n"
        lines.insert(i, debug_line)

        # Also add debug inside the loop
        if i+1 < len(lines):
            loop_indent = indent + 4
            lines.insert(i+2, ' ' * loop_indent + "print(f'DEBUG: Entered training loop - epoch {epoch}')\n")

        modified = True
        print(f"Added debug around training loop at line {i+1}")

    # Check if there's an early exit condition
    if 'if epochs' in line or 'if start_epoch' in line:
        if 'return' in lines[i:i+5]:  # Check next 5 lines for return
            print(f"WARNING: Found potential early exit condition at line {i+1}: {line.strip()}")

# Write the modified file
if modified:
    with open(train_file, 'w') as f:
        f.writelines(lines)
    print("\n✓ Applied debug patches")
else:
    print("\n⚠ No modifications made (might already be patched)")

# Also check what load_checkpoint returns
print("\n" + "=" * 70)
print("Checking load_checkpoint function...")

# Look in models.py
models_file = 'StyleTTS2/models.py'
if os.path.exists(models_file):
    with open(models_file, 'r') as f:
        content = f.read()

    if 'def load_checkpoint' in content:
        print("Found load_checkpoint in models.py")

        # Check what it returns
        import re
        match = re.search(r'def load_checkpoint.*?return\s+([^\\n]+)', content, re.DOTALL)
        if match:
            returns = match.group(1)
            print(f"  Returns: {returns}")

            # Check if start_epoch might be set wrong
            if 'epoch' in returns:
                print("  ⚠ Returns epoch value - check if it's being set correctly")

print("\n" + "=" * 70)
print("\nNow run: python train_gpu.py")
print("Look for DEBUG output to see what values are being used")