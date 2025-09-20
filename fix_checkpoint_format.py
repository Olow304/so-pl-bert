#!/usr/bin/env python
"""
Fix the checkpoint format to match what StyleTTS2 expects.
"""
import torch
import os

def convert_checkpoint_format():
    """Convert checkpoint to expected format."""

    original_path = "Models/LibriTTS/epochs_2nd_00020.pth"
    output_path = "Models/LibriTTS/epochs_2nd_00020_fixed.pth"

    print(f"Loading checkpoint from {original_path}")
    checkpoint = torch.load(original_path, map_location='cpu', weights_only=False)

    # Check structure
    if 'net' in checkpoint:
        print("Found 'net' key, converting to 'model' format")
        new_checkpoint = {
            'model': checkpoint['net'],
            'epoch': 100,  # Default epoch
            'iters': 0
        }
    else:
        print("Checkpoint already in correct format")
        new_checkpoint = checkpoint

    # Save the fixed checkpoint
    torch.save(new_checkpoint, output_path)
    print(f"Saved fixed checkpoint to {output_path}")

    return output_path

def fix_plbert_checkpoint():
    """Fix the PL-BERT checkpoint format."""

    # Load the model from from_scratch
    model_path = "runs/plbert_so/from_scratch/pytorch_model.bin"
    output_path = "runs/plbert_so/packaged/step_000001.pt"

    if os.path.exists(model_path):
        print(f"Loading PL-BERT model from {model_path}")
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

        # Make sure it's just the state dict
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()

        torch.save(state_dict, output_path)
        print(f"Saved PL-BERT state dict to {output_path}")

if __name__ == "__main__":
    # Fix both checkpoints
    fixed_path = convert_checkpoint_format()
    fix_plbert_checkpoint()

    print(f"\nUse this checkpoint for fine-tuning: {fixed_path}")
    print("Update your config to use this checkpoint")