#!/usr/bin/env python
"""
GPU training script that properly launches StyleTTS2 fine-tuning.
"""
import os
import sys
import torch
import subprocess
import yaml

print("=" * 70)
print("StyleTTS2 Fine-tuning with Somali PL-BERT")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Set environment for better GPU performance
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

print("\nChecking prerequisites...")

# Check data
if os.path.exists('data_styletts2/train_list.txt'):
    with open('data_styletts2/train_list.txt', 'r') as f:
        num_samples = len(f.readlines())
    print(f"✓ Found {num_samples} training samples")
else:
    print("✗ Training data not found!")
    sys.exit(1)

# Check PL-BERT and its checkpoint
plbert_dir = 'runs/plbert_so/packaged'
if os.path.exists(plbert_dir):
    print("✓ Somali PL-BERT directory found")
    # Check for actual checkpoint file
    all_files = os.listdir(plbert_dir)
    print(f"  Files in directory: {all_files}")

    # Create a proper checkpoint if needed
    if 'pytorch_model.bin' in all_files and 'step_000001.pt' not in all_files:
        print("  Converting pytorch_model.bin to step_000001.pt format...")
        import torch
        # Load the model weights
        model_path = os.path.join(plbert_dir, 'pytorch_model.bin')
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

        # Save in the expected format for StyleTTS2
        checkpoint = {
            'model': state_dict,
            'iteration': 1,
            'optimizer': None,
            'learning_rate': 1e-5,
            'config': {}
        }
        checkpoint_path = os.path.join(plbert_dir, 'step_000001.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"  ✓ Created {checkpoint_path}")
    elif 'step_000001.pt' in all_files:
        print("  ✓ Found step_000001.pt checkpoint")
else:
    print("✗ Somali PL-BERT not found!")
    sys.exit(1)

# Check pretrained model
if os.path.exists('Models/LibriTTS/epochs_2nd_00020_fixed.pth'):
    print("✓ Pretrained StyleTTS2 model found")
else:
    print("✗ Pretrained model not found!")
    print("  Please ensure Models/LibriTTS/epochs_2nd_00020_fixed.pth exists")

# Check and update config to ensure paths are correct
config_path = 'data_styletts2/config_somali_ft.yml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Update PLBERT path to point to the actual model file
if 'pytorch_model.bin' in os.listdir(plbert_dir):
    # Need to use the packaged directory with the bin file
    config['PLBERT_dir'] = os.path.abspath(plbert_dir)
    print(f"\n✓ Updated PLBERT_dir to: {config['PLBERT_dir']}")

    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("\n" + "=" * 70)
print("Starting training process...")
print("=" * 70)

# Use subprocess to run training with proper output
config_path = os.path.abspath(config_path)

# Change to StyleTTS2 directory and run
os.chdir('StyleTTS2')

cmd = [
    sys.executable,  # Use same Python interpreter
    "train_finetune.py",
    "--config_path", config_path
]

print(f"Command: {' '.join(cmd)}")
print("\nTraining output:")
print("-" * 70)
sys.stdout.flush()

try:
    # Run with real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env={**os.environ, 'PYTHONUNBUFFERED': '1'}  # Force unbuffered output
    )

    # Track if we're actually training
    training_started = False
    epoch_count = 0

    # Print output line by line
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        sys.stdout.flush()

        # Check if training is actually happening
        if 'Epoch' in line or 'epoch' in line:
            training_started = True
            epoch_count += 1
        elif 'loss' in line.lower():
            training_started = True

    # Wait for completion
    process.wait()

    if process.returncode == 0:
        print("\n" + "=" * 70)
        if training_started:
            print(f"Training completed successfully! ({epoch_count} epochs detected)")
        else:
            print("Process completed but no training output detected.")
            print("The script may have exited early. Check:")
            print("  1. Models/Somali/train.log for any logs")
            print("  2. If train_finetune.py requires additional arguments")
            print("  3. If there's a validation step failing silently")
    else:
        print("\n" + "=" * 70)
        print(f"Training exited with code: {process.returncode}")
        print("Check Models/Somali/train.log for details")

except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")
    process.terminate()
    process.wait()
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()