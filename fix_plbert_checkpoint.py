#!/usr/bin/env python
"""
Fix the PL-BERT checkpoint to have the correct format for StyleTTS2.
"""
import os
import torch
import pickle

print("Fixing Somali PL-BERT checkpoint...")

plbert_dir = 'runs/plbert_so/packaged'

# Check what we have
files = os.listdir(plbert_dir)
print(f"Files in {plbert_dir}: {files}")

# Load the existing step_000001.pt to see what's in it
checkpoint_path = os.path.join(plbert_dir, 'step_000001.pt')
if os.path.exists(checkpoint_path):
    print(f"\nLoading {checkpoint_path}...")
    data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"Type of loaded data: {type(data)}")

    if hasattr(data, '__dict__'):
        print(f"Attributes: {list(data.__dict__.keys())[:10]}")  # First 10 attributes

# Look for the actual model file in various locations
print("\nSearching for model weights...")

possible_paths = [
    os.path.join(plbert_dir, 'pytorch_model.bin'),
    'runs/plbert_so/pytorch_model.bin',
    'runs/plbert_so/model.pt',
    'runs/plbert_so/best_model.pt',
    'runs/plbert_so/final_model.pt',
    # Check checkpoint directories
    'runs/plbert_so/checkpoint-1000/pytorch_model.bin',
    'runs/plbert_so/checkpoint-2000/pytorch_model.bin',
    'runs/plbert_so/checkpoint-3000/pytorch_model.bin',
    'runs/plbert_so/checkpoint-4000/pytorch_model.bin',
    'runs/plbert_so/checkpoint-5000/pytorch_model.bin',
]

model_path = None
for path in possible_paths:
    if os.path.exists(path):
        model_path = path
        print(f"Found model at: {model_path}")
        break

# If not found, list what's in the runs directory to help debug
if not model_path:
    print("\nListing contents of runs/plbert_so/:")
    if os.path.exists('runs/plbert_so'):
        for item in os.listdir('runs/plbert_so'):
            item_path = os.path.join('runs/plbert_so', item)
            if os.path.isdir(item_path):
                print(f"  Directory: {item}/")
                # Check inside checkpoint directories
                if 'checkpoint' in item:
                    for subitem in os.listdir(item_path)[:5]:  # First 5 files
                        print(f"    - {subitem}")
            else:
                print(f"  File: {item}")

if model_path and os.path.exists(model_path):
    print(f"\nLoading model from {model_path}...")
    model_state = torch.load(model_path, map_location='cpu', weights_only=False)

    print(f"Model state type: {type(model_state)}")
    if isinstance(model_state, dict):
        print(f"Model state keys: {list(model_state.keys())[:5]}")  # First 5 keys

        # Check if it's already in the right format
        if 'model' in model_state or 'state_dict' in model_state:
            actual_state = model_state.get('model', model_state.get('state_dict'))
        else:
            actual_state = model_state

        # Create a proper checkpoint for StyleTTS2
        proper_checkpoint = {
            'model': actual_state,
            'iteration': 1,
            'optimizer': None,
            'learning_rate': 1e-5
        }

        # Save it
        new_checkpoint_path = os.path.join(plbert_dir, 'step_000001.pt')

        # Backup the old one if it exists
        if os.path.exists(new_checkpoint_path):
            backup_path = new_checkpoint_path + '.bak'
            os.rename(new_checkpoint_path, backup_path)
            print(f"Backed up old checkpoint to {backup_path}")

        torch.save(actual_state, new_checkpoint_path)  # Save just the state dict
        print(f"\n✓ Saved fixed checkpoint to {new_checkpoint_path}")

        # Verify it loads correctly
        test_load = torch.load(new_checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(test_load, dict):
            print(f"✓ Checkpoint verified - contains {len(test_load)} parameters")
            # Show a few parameter names to verify it's the model
            param_names = list(test_load.keys())[:5]
            print(f"  Sample parameters: {param_names}")
    else:
        print(f"Unexpected model state type: {type(model_state)}")
        print("Trying to save as-is...")
        torch.save(model_state, os.path.join(plbert_dir, 'step_000001.pt'))
else:
    print("\n✗ Could not find model weights")
    print("\nLet's check if there's a model.safetensors file:")

    # Check for safetensors format
    safetensor_paths = [
        os.path.join(plbert_dir, 'model.safetensors'),
        'runs/plbert_so/model.safetensors',
        'runs/plbert_so/checkpoint-5000/model.safetensors',
    ]

    for path in safetensor_paths:
        if os.path.exists(path):
            print(f"Found safetensors model at: {path}")
            print("Note: This needs to be converted to .pt format")
            # Try to load with safetensors
            try:
                from safetensors import safe_open
                from safetensors.torch import load_file

                state_dict = load_file(path)
                print(f"Loaded safetensors model with {len(state_dict)} parameters")

                # Save as .pt file
                new_checkpoint_path = os.path.join(plbert_dir, 'step_000001.pt')
                if os.path.exists(new_checkpoint_path):
                    backup_path = new_checkpoint_path + '.bak'
                    os.rename(new_checkpoint_path, backup_path)
                    print(f"Backed up old checkpoint to {backup_path}")

                torch.save(state_dict, new_checkpoint_path)
                print(f"\n✓ Converted and saved model to {new_checkpoint_path}")
                model_path = new_checkpoint_path  # Set this so we don't show error
            except ImportError:
                print("safetensors library not installed. Install with: pip install safetensors")
            break

    if not model_path:
        print("\nCould not find any model weights.")
        print("Please check if the model was properly trained and saved.")

print("\nDone!")