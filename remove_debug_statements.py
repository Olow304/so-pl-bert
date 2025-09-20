#!/usr/bin/env python
"""
Remove debug statements (set_trace) that are blocking training.
"""
import os
import shutil

print("Removing debug statements from training code...")
print("=" * 70)

train_file = 'StyleTTS2/train_finetune.py'

# Backup first
backup_file = train_file + '.debug_statements_backup'
if not os.path.exists(backup_file):
    shutil.copy(train_file, backup_file)
    print(f"Backed up to {backup_file}")

# Read the file
with open(train_file, 'r') as f:
    lines = f.readlines()

print("\nLooking for debug statements...")

# Find and comment out set_trace() calls
modified = False
debug_lines = []

for i, line in enumerate(lines):
    if 'set_trace()' in line and not line.strip().startswith('#'):
        debug_lines.append(i+1)
        # Comment it out
        indent = len(line) - len(line.lstrip())
        lines[i] = ' ' * indent + '# ' + line.lstrip()
        modified = True

    # Also remove or comment out pdb imports
    if 'from pdb import set_trace' in line or 'import pdb' in line:
        if not line.strip().startswith('#'):
            lines[i] = '# ' + line
            modified = True
            print(f"  Commented out pdb import at line {i+1}")

    # Also check for ipdb
    if 'from ipdb import set_trace' in line or 'import ipdb' in line:
        if not line.strip().startswith('#'):
            lines[i] = '# ' + line
            modified = True
            print(f"  Commented out ipdb import at line {i+1}")

if debug_lines:
    print(f"\n  Found and commented out set_trace() at lines: {debug_lines}")

# Write the modified file
if modified:
    with open(train_file, 'w') as f:
        f.writelines(lines)

    print("\n✓ Removed/commented all debug statements")
else:
    print("\n  No debug statements found or already commented")

# Also check for any print statements that might be slowing down training
print("\n" + "=" * 70)
print("Checking for excessive print statements in training loop...")

excessive_prints = []
in_training_loop = False

for i, line in enumerate(lines):
    # Detect if we're in the training loop
    if 'for epoch in' in line:
        in_training_loop = True

    if in_training_loop:
        if 'print(' in line and not line.strip().startswith('#'):
            # Check if it's a progress print (those are OK)
            if 'epoch' not in line.lower() and 'loss' not in line.lower():
                excessive_prints.append(i+1)

if excessive_prints:
    print(f"  Found {len(excessive_prints)} print statements in training loop")
    print("  (These can slow down training but won't stop it)")

print("\n" + "=" * 70)
print("IMPORTANT: If you're currently in the debugger, type these commands:")
print("=" * 70)
print("\n1. Type 'c' and press Enter to continue past the breakpoint")
print("2. Or type 'quit' to exit and restart training")
print("\nThen run: python monitor_training.py")
print("\n✓ Training should now run without interruption!")