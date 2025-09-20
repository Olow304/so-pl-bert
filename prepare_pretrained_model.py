#!/usr/bin/env python
"""
Prepare a pre-trained model checkpoint without BERT weights for fine-tuning.
"""
import torch
import os

def create_pretrained_without_bert():
    """Create a modified checkpoint without BERT weights."""

    # Load the original pre-trained model
    original_path = "Models/LibriTTS/epochs_2nd_00020.pth"
    output_path = "Models/LibriTTS/epochs_2nd_00020_no_bert.pth"

    if not os.path.exists(original_path):
        print(f"Error: Pre-trained model not found at {original_path}")
        return

    print(f"Loading pre-trained model from {original_path}")
    checkpoint = torch.load(original_path, map_location='cpu', weights_only=False)

    # Remove BERT weights from the checkpoint
    if 'model' in checkpoint:
        model_state = checkpoint['model']
    else:
        model_state = checkpoint

    # Create new state dict without BERT weights
    new_model_state = {}
    removed_keys = []

    for key, value in model_state.items():
        # Skip BERT/PLBERT related keys
        if any(x in key.lower() for x in ['bert', 'plbert']):
            removed_keys.append(key)
            continue
        new_model_state[key] = value

    print(f"Removed {len(removed_keys)} BERT-related keys")
    if removed_keys[:5]:
        print(f"First few removed keys: {removed_keys[:5]}")

    # Save the modified checkpoint
    if 'model' in checkpoint:
        checkpoint['model'] = new_model_state
    else:
        checkpoint = new_model_state

    torch.save(checkpoint, output_path)
    print(f"Saved modified checkpoint to {output_path}")

    return output_path

if __name__ == "__main__":
    output = create_pretrained_without_bert()
    if output:
        print(f"\nUse this checkpoint for fine-tuning: {output}")
        print("Update your config to use this checkpoint instead of the original")