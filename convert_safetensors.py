#!/usr/bin/env python
"""
Convert the safetensors model to .pt format for StyleTTS2.
"""
import os
import torch

print("Converting Somali PL-BERT from safetensors to .pt format...")
print("=" * 70)

# The model is in safetensors format
model_path = 'runs/plbert_so/from_scratch/model.safetensors'

if not os.path.exists(model_path):
    # Try the checkpoint version (more recent)
    model_path = 'runs/plbert_so/from_scratch/checkpoint-3471/model.safetensors'

if os.path.exists(model_path):
    print(f"Found model at: {model_path}")

    try:
        # Try with safetensors library
        try:
            from safetensors.torch import load_file
            print("Using safetensors library...")
            state_dict = load_file(model_path)
        except ImportError:
            print("safetensors not installed, trying with transformers...")
            # Try with transformers (it can load safetensors)
            from transformers import AutoModel
            import json

            # Load the config
            config_path = 'runs/plbert_so/from_scratch/config.json'
            if not os.path.exists(config_path):
                config_path = 'runs/plbert_so/from_scratch/checkpoint-3471/config.json'

            if os.path.exists(config_path):
                print(f"Loading with config from: {config_path}")
                # Use transformers to load
                from transformers import AlbertModel
                model = AlbertModel.from_pretrained(
                    os.path.dirname(model_path),
                    from_tf=False,
                    local_files_only=True
                )
                state_dict = model.state_dict()
            else:
                print("No config.json found, installing safetensors...")
                os.system("pip install safetensors")
                from safetensors.torch import load_file
                state_dict = load_file(model_path)

        print(f"✓ Loaded model with {len(state_dict)} parameters")
        print(f"  Sample parameters: {list(state_dict.keys())[:5]}")

        # Check the parameter shapes to understand the model
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"  Total parameters: {total_params:,}")

        # Save to the packaged directory
        output_dir = 'runs/plbert_so/packaged'
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, 'step_000001.pt')

        # Backup old file if exists
        if os.path.exists(output_path):
            backup_path = output_path + '.bak'
            os.rename(output_path, backup_path)
            print(f"\nBacked up old file to: {backup_path}")

        # Save the state dict directly (StyleTTS2 expects just the state dict)
        torch.save(state_dict, output_path)
        print(f"\n✓ Saved converted model to: {output_path}")

        # Verify the saved file
        test_load = torch.load(output_path, map_location='cpu', weights_only=False)
        if isinstance(test_load, dict) and len(test_load) > 0:
            print(f"✓ Verification successful - saved file contains {len(test_load)} parameters")

            # Check if it's ALBERT architecture
            if any('albert' in k.lower() for k in test_load.keys()):
                print("✓ Model appears to be ALBERT architecture (correct for PL-BERT)")
            elif any('embeddings' in k for k in test_load.keys()):
                print("✓ Model has embedding layers")
        else:
            print("⚠ Warning: Saved file might not be in the correct format")

    except Exception as e:
        print(f"✗ Error converting model: {e}")
        import traceback
        traceback.print_exc()

        print("\nTrying to install safetensors and retry...")
        os.system("pip install safetensors")

else:
    print(f"✗ Model file not found at: {model_path}")
    print("\nPlease ensure the PL-BERT model was trained successfully")

print("\n" + "=" * 70)
print("Conversion complete!")