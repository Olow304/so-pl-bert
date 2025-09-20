#!/usr/bin/env python
"""
Run StyleTTS2 training directly.
"""
import os
import sys
import subprocess

print("=" * 70)
print("Running StyleTTS2 Training Directly")
print("=" * 70)

# The issue is that train_finetune.py uses click and expects to be run as a script
# Let's look at what it actually does

train_file = 'StyleTTS2/train_finetune.py'
print(f"\n1. Checking {train_file} structure...")

with open(train_file, 'r') as f:
    lines = f.readlines()

# Find the click command function
main_func_name = None
for i, line in enumerate(lines):
    if '@click.command()' in line:
        # The next function def is the main function
        for j in range(i+1, min(i+10, len(lines))):
            if 'def ' in lines[j]:
                import re
                match = re.search(r'def\s+(\w+)\s*\(', lines[j])
                if match:
                    main_func_name = match.group(1)
                    print(f"   Found main function: {main_func_name}")
                    break
        break

if not main_func_name:
    print("   Could not find main function")
    # Look for if __name__ == '__main__'
    for i, line in enumerate(lines):
        if "__name__ == '__main__'" in line:
            print("   Found __main__ block at line", i+1)
            # Show what's in the main block
            for j in range(i+1, min(i+5, len(lines))):
                print(f"     Line {j+1}: {lines[j].rstrip()}")

print("\n2. Running training with proper click invocation...")
print("=" * 70)

# Change to StyleTTS2 directory and run the training script directly
os.chdir('StyleTTS2')

config_path = os.path.abspath('../data_styletts2/config_somali_ft.yml')

# Run the training script as it expects to be run
cmd = [
    sys.executable,
    'train_finetune.py',
    '--config_path', config_path
]

print(f"Command: {' '.join(cmd)}")
print("\nStarting training...")
print("-" * 70)

# Set environment to see more output
env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'
env['CUDA_LAUNCH_BLOCKING'] = '1'

try:
    # Run with subprocess to capture all output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=env
    )

    # Read output line by line
    output_lines = []
    training_started = False

    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        output_lines.append(line)

        # Check if training actually started
        if 'epoch' in line.lower() or 'step' in line.lower() or 'loss' in line.lower():
            training_started = True

    process.wait()

    print("-" * 70)

    if process.returncode == 0:
        if training_started:
            print("\n✓ Training completed successfully!")
        else:
            print("\n⚠ Process completed but no training output detected")
            print("\nLast 10 lines of output:")
            for line in output_lines[-10:]:
                print(f"  {line.rstrip()}")
    else:
        print(f"\n✗ Training exited with code: {process.returncode}")

        # Check if there's a specific error
        error_lines = [l for l in output_lines if 'error' in l.lower() or 'exception' in l.lower()]
        if error_lines:
            print("\nErrors found:")
            for line in error_lines[:5]:
                print(f"  {line.rstrip()}")

except KeyboardInterrupt:
    print("\n\nInterrupted by user")
    process.terminate()
except Exception as e:
    print(f"\n✗ Error: {e}")

print("\n" + "=" * 70)

# Check for any log files
log_dir = '../Models/Somali'
if os.path.exists(log_dir):
    print(f"\nChecking {log_dir}:")
    for file in os.listdir(log_dir):
        file_path = os.path.join(log_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"  {file}: {size} bytes")

            if file == 'train.log' and size > 0:
                print("    Contents:")
                with open(file_path, 'r') as f:
                    content = f.read()
                    if content:
                        print(f"    {content[:500]}")  # First 500 chars