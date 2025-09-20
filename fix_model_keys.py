#!/usr/bin/env python
"""
Fix the model key names to match what StyleTTS2 expects.
"""
import torch
import os

print("Fixing PL-BERT model key names...")
print("=" * 70)

# Load the current model
model_path = 'runs/plbert_so/packaged/step_000001.pt'

if os.path.exists(model_path):
    print(f"Loading model from: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

    print(f"Current model has {len(state_dict)} parameters")
    print("Sample keys (current):")
    for i, key in enumerate(list(state_dict.keys())[:5]):
        print(f"  {key}")

    # Check if keys have 'albert.' prefix
    if any(k.startswith('albert.') for k in state_dict.keys()):
        print("\nRemoving 'albert.' prefix from keys...")

        # Remove the 'albert.' prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('albert.'):
                new_key = key[7:]  # Remove 'albert.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = value

        print(f"\nNew model has {len(new_state_dict)} parameters")
        print("Sample keys (fixed):")
        for i, key in enumerate(list(new_state_dict.keys())[:5]):
            print(f"  {key}")

        # Backup the old file
        backup_path = model_path + '.with_prefix'
        os.rename(model_path, backup_path)
        print(f"\nBacked up original to: {backup_path}")

        # Save the fixed model
        torch.save(new_state_dict, model_path)
        print(f"Saved fixed model to: {model_path}")

    # Also check for 'predictions' keys that might not be needed
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    if any('predictions' in k for k in state_dict.keys()):
        print("\nRemoving 'predictions' layer (not needed for PL-BERT in StyleTTS2)...")

        filtered_dict = {k: v for k, v in state_dict.items() if 'predictions' not in k}

        print(f"Filtered model has {len(filtered_dict)} parameters (was {len(state_dict)})")

        # Save the filtered model
        torch.save(filtered_dict, model_path)
        print(f"Saved filtered model to: {model_path}")

    # Final verification
    final_state = torch.load(model_path, map_location='cpu', weights_only=False)
    print("\n" + "=" * 70)
    print("Final model structure:")
    print(f"  Total parameters: {len(final_state)}")

    # Group keys by component
    embeddings_keys = [k for k in final_state.keys() if 'embeddings' in k]
    encoder_keys = [k for k in final_state.keys() if 'encoder' in k]
    pooler_keys = [k for k in final_state.keys() if 'pooler' in k]
    other_keys = [k for k in final_state.keys() if k not in embeddings_keys + encoder_keys + pooler_keys]

    print(f"  Embedding layers: {len(embeddings_keys)}")
    print(f"  Encoder layers: {len(encoder_keys)}")
    print(f"  Pooler layers: {len(pooler_keys)}")
    if other_keys:
        print(f"  Other layers: {len(other_keys)}")
        print(f"    {other_keys[:5]}")

    print("\n✓ Model keys fixed and ready for StyleTTS2")

else:
    print(f"✗ Model file not found: {model_path}")

print("=" * 70)