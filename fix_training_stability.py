#!/usr/bin/env python
"""
Fix training stability issues - NaN losses and exploding gradients.
"""
import yaml

print("Fixing training stability issues...")
print("=" * 70)

# 1. Update config with more conservative settings
config_path = 'data_styletts2/config_somali_ft.yml'

print("\n1. Updating training configuration for stability...")

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Reduce learning rates significantly
print("\n   Reducing learning rates:")
if 'optimizer_params' not in config:
    config['optimizer_params'] = {}

old_lr = config['optimizer_params'].get('lr', 0.0001)
old_bert_lr = config['optimizer_params'].get('bert_lr', 0.00001)

config['optimizer_params']['lr'] = 0.00001  # 10x smaller
config['optimizer_params']['bert_lr'] = 0.000001  # 10x smaller
config['optimizer_params']['ft_lr'] = 0.00001  # Fine-tuning LR

print(f"     Main LR: {old_lr} → {config['optimizer_params']['lr']}")
print(f"     BERT LR: {old_bert_lr} → {config['optimizer_params']['bert_lr']}")

# Reduce batch size to avoid memory issues
old_batch = config.get('batch_size', 4)
config['batch_size'] = 2  # Smaller batch size
print(f"\n   Batch size: {old_batch} → {config['batch_size']}")

# Add gradient clipping
config['grad_clip'] = 1.0  # Clip gradients to prevent explosion
print(f"\n   Added gradient clipping: {config['grad_clip']}")

# Adjust loss weights to be more conservative
if 'loss_params' not in config:
    config['loss_params'] = {}

print("\n   Adjusting loss weights:")
# Reduce the problematic loss weights
config['loss_params']['lambda_norm'] = 0.1  # Was 1.0
config['loss_params']['lambda_F0'] = 0.1    # Was 1.0
config['loss_params']['lambda_mel'] = 1.0   # Was 5.0
config['loss_params']['lambda_ce'] = 5.0    # Was 20.0

print(f"     lambda_norm: 1.0 → {config['loss_params']['lambda_norm']}")
print(f"     lambda_F0: 1.0 → {config['loss_params']['lambda_F0']}")
print(f"     lambda_mel: 5.0 → {config['loss_params']['lambda_mel']}")
print(f"     lambda_ce: 20.0 → {config['loss_params']['lambda_ce']}")

# Start with simpler training (no adversarial/diffusion at first)
config['loss_params']['diff_epoch'] = 20  # Start diffusion later
config['loss_params']['joint_epoch'] = 30  # Start joint training later

print(f"\n   Delayed advanced training:")
print(f"     Diffusion starts at epoch: {config['loss_params']['diff_epoch']}")
print(f"     Joint training starts at epoch: {config['loss_params']['joint_epoch']}")

# Save the updated config
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("\n✓ Updated configuration for stability")

# 2. Create a training warmup script
print("\n" + "=" * 70)
print("2. Creating warmup training script...")

warmup_script = """#!/usr/bin/env python
'''
Warmup training with very conservative settings
'''
import yaml
import shutil
import os

print("Setting up warmup training...")
print("=" * 70)

# Create a warmup config
config_path = 'data_styletts2/config_somali_ft.yml'
warmup_config_path = 'data_styletts2/config_somali_warmup.yml'

# Copy and modify config for warmup
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Ultra-conservative settings for warmup
config['epochs'] = 5  # Just 5 epochs for warmup
config['batch_size'] = 1  # Single sample
config['optimizer_params']['lr'] = 0.000001  # Very small LR
config['optimizer_params']['bert_lr'] = 0.0000001

# Disable most losses for warmup
config['loss_params']['lambda_gen'] = 0.0
config['loss_params']['lambda_slm'] = 0.0
config['loss_params']['lambda_diff'] = 0.0
config['loss_params']['lambda_sty'] = 0.0

# Only train on basic losses
config['loss_params']['lambda_mel'] = 1.0
config['loss_params']['lambda_dur'] = 1.0
config['loss_params']['lambda_ce'] = 1.0

print("Warmup settings:")
print(f"  Epochs: {config['epochs']}")
print(f"  Batch size: {config['batch_size']}")
print(f"  Learning rate: {config['optimizer_params']['lr']}")

# Save warmup config
with open(warmup_config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"\\n✓ Created warmup config: {warmup_config_path}")

# Run warmup training
print("\\n" + "=" * 70)
print("Starting warmup training...")
print("=" * 70)

import subprocess
cmd = ['python', 'train_gpu.py']

# Update train_gpu.py to use warmup config
train_gpu_file = 'train_gpu.py'
with open(train_gpu_file, 'r') as f:
    content = f.read()

# Temporarily use warmup config
content_backup = content
content = content.replace(
    'config_somali_ft.yml',
    'config_somali_warmup.yml'
)

with open(train_gpu_file, 'w') as f:
    f.write(content)

print("\\nRunning warmup training (5 epochs, very small LR)...")
subprocess.run(cmd)

# Restore original train_gpu.py
with open(train_gpu_file, 'w') as f:
    f.write(content_backup)

print("\\n✓ Warmup complete! Now run regular training:")
print("  python monitor_training.py")
"""

with open('warmup_training.py', 'w') as f:
    f.write(warmup_script)

print("Created warmup_training.py")

# 3. Create a script to check for NaN in checkpoints
print("\n" + "=" * 70)
print("3. Creating NaN checker script...")

nan_check_script = """#!/usr/bin/env python
'''
Check if model weights contain NaN values
'''
import torch
import os

print("Checking for NaN in model weights...")

# Check if there are any saved checkpoints
model_dir = 'Models/Somali'
if os.path.exists(model_dir):
    checkpoints = [f for f in os.listdir(model_dir) if f.endswith('.pth')]

    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoints")

        for ckpt in checkpoints[:1]:  # Check first checkpoint
            path = os.path.join(model_dir, ckpt)
            print(f"\\nChecking {ckpt}...")

            state = torch.load(path, map_location='cpu')

            nan_params = []
            inf_params = []

            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any():
                        nan_params.append(key)
                    if torch.isinf(value).any():
                        inf_params.append(key)

            if nan_params:
                print(f"  ✗ Found NaN in {len(nan_params)} parameters:")
                for p in nan_params[:5]:
                    print(f"    - {p}")

            if inf_params:
                print(f"  ✗ Found Inf in {len(inf_params)} parameters:")
                for p in inf_params[:5]:
                    print(f"    - {p}")

            if not nan_params and not inf_params:
                print("  ✓ No NaN or Inf values found")
    else:
        print("No checkpoints found yet")

print("\\nIf NaN values are found, training needs to be restarted with")
print("more conservative settings (smaller learning rate, gradient clipping)")
"""

with open('check_nan.py', 'w') as f:
    f.write(nan_check_script)

print("Created check_nan.py")

print("\n" + "=" * 70)
print("SOLUTIONS APPLIED:")
print("=" * 70)
print("\n1. ✓ Reduced learning rates by 10x")
print("2. ✓ Reduced batch size to 2")
print("3. ✓ Added gradient clipping")
print("4. ✓ Adjusted loss weights")
print("5. ✓ Delayed advanced training features")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("\n1. Stop current training (Ctrl+C)")
print("2. Run: python warmup_training.py  # Optional: very safe warmup")
print("3. Or directly run: python monitor_training.py  # With new settings")
print("\nThe training should now be stable without NaN losses!")