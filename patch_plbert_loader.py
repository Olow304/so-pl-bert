#!/usr/bin/env python
"""
Patch StyleTTS2's PLBERT loader to work with our Somali model.
"""
import os

def patch_plbert_util():
    """Patch the PLBERT util.py to handle our model format."""

    util_file = "StyleTTS2/Utils/PLBERT/util.py"

    # Create a new version that handles our model
    new_content = '''
import torch
import yaml
from transformers import AlbertConfig, AlbertModel
import os

def load_plbert(log_dir):
    """Load PL-BERT model - adapted for Somali PL-BERT."""

    # Try to load config
    config_path = os.path.join(log_dir, "config.yml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            plbert_config = yaml.safe_load(f)
            if 'model_params' in plbert_config:
                albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
            else:
                # Fallback for our Somali model
                albert_base_configuration = AlbertConfig(
                    vocab_size=116,
                    hidden_size=512,
                    num_hidden_layers=6,
                    num_attention_heads=8,
                    intermediate_size=2048,
                    hidden_act="gelu_new",
                    hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1,
                    max_position_embeddings=256,
                    type_vocab_size=1,
                    initializer_range=0.02,
                    layer_norm_eps=1e-12,
                    embedding_size=128,
                )
    else:
        # Use default config for Somali model
        albert_base_configuration = AlbertConfig(
            vocab_size=116,
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048,
            hidden_act="gelu_new",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=256,
            type_vocab_size=1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            embedding_size=128,
        )

    # Create model
    bert = AlbertModel(albert_base_configuration)

    # Try to load checkpoint
    checkpoint_path = os.path.join(log_dir, "step_000001.pt")
    if not os.path.exists(checkpoint_path):
        # Try alternative paths
        for alt_path in ["pytorch_model.bin", "model.pt", "best_model.pt"]:
            full_path = os.path.join(log_dir, alt_path)
            if os.path.exists(full_path):
                checkpoint_path = full_path
                break

    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    bert.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    bert.load_state_dict(checkpoint['state_dict'])
                else:
                    # Try to load directly
                    bert.load_state_dict(checkpoint)
            else:
                # Assume it's the state dict directly
                bert.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Warning: Could not load checkpoint from {checkpoint_path}: {e}")
            print("Using randomly initialized model")
    else:
        # Try to load from HuggingFace format
        try:
            from transformers import AlbertModel as HFAlbertModel
            bert = HFAlbertModel.from_pretrained(log_dir)
        except:
            print(f"Warning: No checkpoint found in {log_dir}")
            print("Using randomly initialized model")

    return bert
'''

    # Backup original
    if os.path.exists(util_file):
        import shutil
        backup_file = util_file + ".backup"
        if not os.path.exists(backup_file):
            shutil.copy(util_file, backup_file)
            print(f"Backed up original to {backup_file}")

    # Write new version
    with open(util_file, 'w') as f:
        f.write(new_content)

    print(f"Patched {util_file}")

if __name__ == "__main__":
    print("Patching StyleTTS2 PLBERT loader...")
    patch_plbert_util()
    print("Patching complete!")