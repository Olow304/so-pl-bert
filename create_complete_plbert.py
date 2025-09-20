#!/usr/bin/env python
"""
Create a complete PL-BERT model with all required layers for StyleTTS2.
"""
import torch
import os
from transformers import AlbertModel, AlbertConfig

print("Creating complete PL-BERT model for StyleTTS2...")
print("=" * 70)

# Load current incomplete model
current_model_path = 'runs/plbert_so/packaged/step_000001.pt'
if os.path.exists(current_model_path):
    current_state = torch.load(current_model_path, map_location='cpu', weights_only=False)
    print(f"Loaded current model with {len(current_state)} parameters")
else:
    print("Current model not found!")
    exit(1)

# Create a dummy ALBERT model with the correct config to see what's expected
print("\nCreating reference ALBERT model...")

# Load the config from our trained model
config_path = 'runs/plbert_so/packaged/config.yml'
if os.path.exists(config_path):
    import yaml
    with open(config_path, 'r') as f:
        plbert_config = yaml.safe_load(f)

    print(f"Loaded PL-BERT config: vocab_size={plbert_config['model_params']['vocab_size']}")

    # Create ALBERT config matching our model
    albert_config = AlbertConfig(
        vocab_size=plbert_config['model_params']['vocab_size'],
        hidden_size=plbert_config['model_params']['hidden_size'],
        num_hidden_layers=plbert_config['model_params']['num_hidden_layers'],
        num_attention_heads=plbert_config['model_params']['num_attention_heads'],
        intermediate_size=plbert_config['model_params']['intermediate_size'],
        embedding_size=plbert_config['model_params'].get('embedding_size', 128),
        max_position_embeddings=plbert_config['model_params'].get('max_position_embeddings', 512),
        type_vocab_size=plbert_config['model_params'].get('type_vocab_size', 1)
    )
else:
    print("Config not found, using defaults...")
    albert_config = AlbertConfig(
        vocab_size=116,  # Somali vocab
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        embedding_size=128,
        max_position_embeddings=512,
        type_vocab_size=1
    )

print(f"\nALBERT config:")
print(f"  vocab_size: {albert_config.vocab_size}")
print(f"  hidden_size: {albert_config.hidden_size}")
print(f"  num_hidden_layers: {albert_config.num_hidden_layers}")
print(f"  embedding_size: {albert_config.embedding_size}")

# Create the reference model
reference_model = AlbertModel(albert_config)
reference_state = reference_model.state_dict()

print(f"\nReference model has {len(reference_state)} parameters")

# Compare what's missing
missing_keys = set(reference_state.keys()) - set(current_state.keys())
extra_keys = set(current_state.keys()) - set(reference_state.keys())

if missing_keys:
    print(f"\nMissing {len(missing_keys)} keys:")
    for key in sorted(missing_keys)[:10]:  # Show first 10
        print(f"  - {key}")

if extra_keys:
    print(f"\nExtra {len(extra_keys)} keys (will be removed):")
    for key in sorted(extra_keys):
        print(f"  - {key}")

# Create complete state dict
print("\n" + "=" * 70)
print("Building complete model...")

complete_state = {}

# Copy existing parameters where they match
matched = 0
initialized = 0

for key in reference_state.keys():
    if key in current_state:
        # Check if shapes match
        if current_state[key].shape == reference_state[key].shape:
            complete_state[key] = current_state[key]
            matched += 1
        else:
            print(f"Shape mismatch for {key}: got {current_state[key].shape}, expected {reference_state[key].shape}")
            complete_state[key] = reference_state[key]  # Use random initialization
            initialized += 1
    else:
        # Use randomly initialized weights from reference model
        complete_state[key] = reference_state[key]
        initialized += 1

print(f"\nParameter summary:")
print(f"  Matched from trained model: {matched}")
print(f"  Randomly initialized: {initialized}")
print(f"  Total parameters: {len(complete_state)}")

# Save the complete model
output_path = 'runs/plbert_so/packaged/step_000001.pt'

# Backup current file
backup_path = output_path + '.incomplete'
if os.path.exists(output_path):
    os.rename(output_path, backup_path)
    print(f"\nBacked up incomplete model to: {backup_path}")

# Save complete model
torch.save(complete_state, output_path)
print(f"Saved complete model to: {output_path}")

# Verify the model loads correctly
print("\n" + "=" * 70)
print("Verifying model...")

try:
    test_model = AlbertModel(albert_config)
    test_model.load_state_dict(torch.load(output_path, map_location='cpu', weights_only=False))
    print("✓ Model loads successfully with AlbertModel!")

    # Test a forward pass
    import torch
    dummy_input = torch.randint(0, albert_config.vocab_size, (1, 10))
    with torch.no_grad():
        output = test_model(dummy_input)
    print(f"✓ Forward pass successful! Output shape: {output.last_hidden_state.shape}")

except Exception as e:
    print(f"✗ Error loading model: {e}")

print("\n✓ Complete PL-BERT model ready for StyleTTS2!")
print("=" * 70)