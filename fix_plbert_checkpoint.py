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

# Look for the actual model file
model_path = os.path.join(plbert_dir, 'pytorch_model.bin')
if not os.path.exists(model_path):
    # Try to find it in the parent directory
    parent_model_path = 'runs/plbert_so/pytorch_model.bin'
    if os.path.exists(parent_model_path):
        model_path = parent_model_path
        print(f"\nFound model at {model_path}")
    else:
        # Check if there's a checkpoint in the training directory
        training_checkpoints = [
            'runs/plbert_so/checkpoint-1000/pytorch_model.bin',
            'runs/plbert_so/checkpoint-2000/pytorch_model.bin',
            'runs/plbert_so/checkpoint-3000/pytorch_model.bin',
            'runs/plbert_so/checkpoint-4000/pytorch_model.bin',
            'runs/plbert_so/checkpoint-5000/pytorch_model.bin',
        ]

        for ckpt_path in training_checkpoints:
            if os.path.exists(ckpt_path):
                model_path = ckpt_path
                print(f"\nFound model checkpoint at {model_path}")
                break

if os.path.exists(model_path):
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
    print("\n✗ Could not find pytorch_model.bin")
    print("Please ensure the PL-BERT model was properly trained and packaged")

print("\nDone!")