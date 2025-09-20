#!/usr/bin/env python
"""
Fix the TextCleaner print statements and check why training exits.
"""
import os

print("Fixing TextCleaner print statements and investigating training exit...")
print("=" * 70)

# First, fix the print statements in meldataset.py
meldataset_file = 'StyleTTS2/meldataset.py'

with open(meldataset_file, 'r') as f:
    lines = f.readlines()

print("1. Removing TextCleaner print statements...")
fixed = False
for i, line in enumerate(lines):
    # Find the TextCleaner class
    if 'class TextCleaner' in line:
        # Look for print statements in the next 20 lines
        for j in range(i, min(i+20, len(lines))):
            if 'print(text)' in lines[j]:
                print(f"   Found print statement at line {j+1}: {lines[j].strip()}")
                # Comment it out
                lines[j] = lines[j].replace('print(text)', '# print(text)')
                fixed = True

if fixed:
    with open(meldataset_file, 'w') as f:
        f.writelines(lines)
    print("   ✓ Commented out print statements")

# Now check the train_finetune.py for early exits
print("\n2. Checking train_finetune.py for early exits...")

train_file = 'StyleTTS2/train_finetune.py'
with open(train_file, 'r') as f:
    content = f.read()

# Look for click command and main function
if '@click.command()' in content:
    print("   ✓ Uses click for command line")

# Check if there's a proper training loop
if 'for epoch in' in content or 'while epoch' in content:
    print("   ✓ Has training loop")
else:
    print("   ⚠ No obvious training loop found")

# Look for early returns or exits
import re
exits = re.findall(r'^\s*(return|sys\.exit|exit\()', content, re.MULTILINE)
if exits:
    print(f"   Found {len(exits)} potential early exits")

# Check if there's a validation that might be failing
if 'len(train_dataloader) == 0' in content or 'len(dataset) == 0' in content:
    print("   ⚠ Has empty dataset check that might exit")

print("\n3. Creating a minimal training script to bypass click...")
print("=" * 70)

# Create a minimal script that directly runs training
minimal_train = """#!/usr/bin/env python
'''
Minimal training script that bypasses click and runs training directly.
'''
import os
import sys
import torch
import yaml

print("Starting minimal training script...")

# Change to StyleTTS2 directory
os.chdir('StyleTTS2')
sys.path.insert(0, '.')

# Load config
config_path = '../data_styletts2/config_somali_ft.yml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print(f"Loaded config from {config_path}")

# Import training modules
from train_finetune import train_loop
import train_finetune

# Set up the config namespace (mimicking what click would do)
import argparse
args = argparse.Namespace(config_path=config_path)

print("Initializing training...")

# Try to call the main training function directly
try:
    # First check if there's a main function
    if hasattr(train_finetune, 'main'):
        print("Calling main() directly...")
        train_finetune.main(config_path)
    elif hasattr(train_finetune, 'train'):
        print("Calling train() directly...")
        train_finetune.train(config_path)
    else:
        print("No main or train function found, trying to import and run...")

        # Import the entire module and look for the training logic
        import importlib
        spec = importlib.util.spec_from_file_location("train_finetune", "train_finetune.py")
        module = importlib.util.module_from_spec(spec)

        # Simulate click arguments
        import click
        @click.command()
        @click.option('--config_path', type=str, default=config_path)
        def dummy_main(config_path):
            pass

        # Create a context and invoke
        ctx = click.Context(dummy_main)
        ctx.params = {'config_path': config_path}

        # Execute the module with our context
        spec.loader.exec_module(module)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\\nIf training didn't start, there may be an issue with the training script itself.")
"""

with open('minimal_train.py', 'w') as f:
    f.write(minimal_train)

print("Created minimal_train.py")
print("\nNow run: python minimal_train.py")
print("This will try to start training directly without click")