#!/usr/bin/env python
"""
Fix PL-BERT config to match StyleTTS2's expectations.
"""
import yaml
import os
import json

def create_plbert_config():
    """Create a proper config.yml for our Somali PL-BERT."""

    # Read our model's actual config
    config_json_path = "runs/plbert_so/from_scratch/config.json"

    if os.path.exists(config_json_path):
        with open(config_json_path, 'r') as f:
            model_config = json.load(f)
    else:
        # Default ALBERT config based on our training
        model_config = {
            "vocab_size": 116,
            "embedding_size": 128,
            "hidden_size": 512,
            "num_hidden_layers": 6,
            "num_attention_heads": 8,
            "intermediate_size": 2048,
            "max_position_embeddings": 256,
            "type_vocab_size": 1
        }

    # Create StyleTTS2-compatible config
    plbert_config = {
        'model_params': {
            'vocab_size': model_config.get('vocab_size', 116),
            'hidden_size': model_config.get('hidden_size', 512),
            'num_attention_heads': model_config.get('num_attention_heads', 8),
            'intermediate_size': model_config.get('intermediate_size', 2048),
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': model_config.get('max_position_embeddings', 256),
            'hidden_dropout_prob': 0.1,
            'type_vocab_size': 1,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'num_hidden_layers': model_config.get('num_hidden_layers', 6),
            'embedding_size': model_config.get('embedding_size', 128),
            'pad_token_id': 0,
            'bos_token_id': None,
            'eos_token_id': None,
            'inner_group_num': 1,
            'hidden_act': 'gelu_new',
            'classifier_dropout_prob': 0.1,
            'position_embedding_type': 'absolute'
        },
        'preprocess_params': {
            'sr': 24000
        }
    }

    # Save to packaged directory
    config_path = "runs/plbert_so/packaged/config.yml"
    with open(config_path, 'w') as f:
        yaml.dump(plbert_config, f, default_flow_style=False)

    print(f"Created PL-BERT config at {config_path}")

    # Also need to ensure the model checkpoint is in the right format
    import torch

    # Load our checkpoint
    checkpoint_path = "runs/plbert_so/packaged/step_000001.pt"
    if not os.path.exists(checkpoint_path):
        # Try to find the actual model file
        model_path = "runs/plbert_so/from_scratch/pytorch_model.bin"
        if os.path.exists(model_path):
            print(f"Copying model from {model_path} to {checkpoint_path}")
            import shutil
            shutil.copy(model_path, checkpoint_path)

    # Also create the expected util.py in our packaged directory
    util_content = '''
import torch
import pickle

def load_plbert(ckpt_path):
    """Load PL-BERT model for StyleTTS2."""
    checkpoint = torch.load(ckpt_path, weights_only=False)
    return checkpoint

def load_config(config_path):
    """Load config file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
'''

    util_path = "runs/plbert_so/packaged/util.py"
    with open(util_path, 'w') as f:
        f.write(util_content)

    print(f"Created util.py at {util_path}")

if __name__ == "__main__":
    create_plbert_config()
    print("PL-BERT configuration fixed for StyleTTS2 compatibility")