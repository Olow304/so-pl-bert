#!/usr/bin/env python
"""
Check what's in the pre-trained checkpoint.
"""
import torch

def check_checkpoint():
    """Examine the checkpoint structure."""

    checkpoint_path = "Models/LibriTTS/epochs_2nd_00020.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print("Checkpoint keys:", checkpoint.keys() if isinstance(checkpoint, dict) else "Not a dict")

    if isinstance(checkpoint, dict):
        for key in checkpoint.keys():
            if key == 'model':
                model_state = checkpoint['model']
                print(f"\nModel state dict has {len(model_state)} keys")

                # Check for BERT keys
                bert_keys = [k for k in model_state.keys() if 'bert' in k.lower()]
                print(f"Found {len(bert_keys)} BERT-related keys")

                # Show first few keys
                all_keys = list(model_state.keys())
                print("\nFirst 20 model keys:")
                for k in all_keys[:20]:
                    print(f"  {k}")

                # Look for the bert model specifically
                bert_model_keys = [k for k in all_keys if k.startswith('bert.') or k.startswith('module.bert.')]
                if bert_model_keys:
                    print(f"\nFound {len(bert_model_keys)} keys starting with 'bert.'")
                    print("First few:", bert_model_keys[:5])
            elif key == 'optimizer':
                print(f"Optimizer state found")
            else:
                print(f"Key '{key}': {type(checkpoint[key])}")

if __name__ == "__main__":
    check_checkpoint()