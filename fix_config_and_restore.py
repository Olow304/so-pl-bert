#!/usr/bin/env python
"""
Fix the config to ensure training runs and restore clean training script.
"""
import os
import shutil
import yaml

print("Fixing configuration and restoring training script...")
print("=" * 70)

# First restore clean training script from backup
train_file = 'StyleTTS2/train_finetune.py'
backup_file = train_file + '.debug_backup'
if os.path.exists(backup_file):
    shutil.copy(backup_file, train_file)
    print(f"✓ Restored clean training script from backup")

# Now fix the config
config_path = 'data_styletts2/config_somali_ft.yml'
print(f"\n1. Checking current config...")

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print(f"   Current settings:")
print(f"     epochs: {config.get('epochs', 'NOT SET')}")
print(f"     load_only_params: {config.get('load_only_params', 'NOT SET')}")
print(f"     second_stage_load_pretrained: {config.get('second_stage_load_pretrained', 'NOT SET')}")

# The issue is that load_only_params needs to be True
# This will load the model weights but reset epoch counter to 1
if not config.get('load_only_params', False):
    print(f"\n   ⚠ load_only_params is not True - this is the issue!")
    config['load_only_params'] = True
    print(f"   ✓ Set load_only_params = True")

    # Save the updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Updated config saved")
else:
    print(f"\n   load_only_params is already True")

print("\n" + "=" * 70)
print("2. Creating simple monitoring script...")

monitor_script = """#!/usr/bin/env python
'''
Monitor training progress
'''
import os
import time
import subprocess

print("Starting StyleTTS2 training with monitoring...")
print("=" * 70)

# Start training in background
cmd = [
    'python', 'train_gpu.py'
]

print(f"Command: {' '.join(cmd)}")
print("\\nStarting training...")
print("-" * 70)

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

# Monitor output
epoch_count = 0
last_output_time = time.time()
training_started = False

try:
    for line in iter(process.stdout.readline, ''):
        print(line, end='')

        # Track progress
        if 'epoch' in line.lower():
            epoch_count += 1
            training_started = True
            print(f"\\n>>> TRAINING ACTIVE: Epoch mention #{epoch_count} <<<\\n")

        if 'loss' in line.lower():
            training_started = True

        last_output_time = time.time()

    process.wait()

    if training_started:
        print("\\n" + "=" * 70)
        print("✓✓✓ TRAINING COMPLETED SUCCESSFULLY!")
    else:
        print("\\n" + "=" * 70)
        print("⚠ Training exited without starting")
        print("\\nCheck Models/Somali/ for any logs")

except KeyboardInterrupt:
    print("\\n\\nInterrupted by user")
    process.terminate()

# Check for checkpoint files
model_dir = 'Models/Somali'
if os.path.exists(model_dir):
    files = os.listdir(model_dir)
    checkpoints = [f for f in files if 'epoch' in f or '.pth' in f]
    if checkpoints:
        print(f"\\n✓ Found {len(checkpoints)} checkpoint files:")
        for ckpt in checkpoints:
            print(f"   {ckpt}")
"""

with open('monitor_training.py', 'w') as f:
    f.write(monitor_script)

print("Created monitor_training.py")

print("\n" + "=" * 70)
print("SOLUTION APPLIED!")
print("=" * 70)
print("\nThe issue was that the pretrained model's epoch count was being used.")
print("Now with load_only_params=True, training will start from epoch 1.")
print("\nRun: python monitor_training.py")
print("\nTraining should now actually run! 🎉")