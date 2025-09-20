#!/usr/bin/env python
"""
Package a trained PL‑BERT model for use with StyleTTS2.  Copies the model
weights, configuration and token maps into a target directory with names
expected by the StyleTTS2 codebase.  A minimal `config.yml` and `util.py`
are generated to mirror the format used by the official PL‑BERT release.

Example:

    python training/pack.py \
      --input_dir runs/plbert_so/continue \
      --token_maps phonemize/token_maps.pkl \
      --output_dir runs/plbert_so/packaged

"""
import argparse
import json
import logging
import os
import shutil
import pickle


def main():
    parser = argparse.ArgumentParser(description="Package a PL‑BERT model for StyleTTS2.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the trained PL‑BERT model.")
    parser.add_argument("--token_maps", type=str, required=True, help="Pickled token map.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write packaged model.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)
    # Copy model weight file (pytorch_model.bin) as step_000001.pt
    # or .t7 to mimic Torch; we simply copy the binary.
    src_weights = os.path.join(args.input_dir, "pytorch_model.bin")
    if not os.path.exists(src_weights):
        # try HuggingFace safe tensor file
        for fname in os.listdir(args.input_dir):
            if fname.endswith(".bin"):
                src_weights = os.path.join(args.input_dir, fname)
                break
    dst_weights = os.path.join(args.output_dir, "step_000001.pt")
    shutil.copy(src_weights, dst_weights)
    # Copy config.json to config.yml (YAML format expected by StyleTTS2)
    src_config = os.path.join(args.input_dir, "config.json")
    with open(src_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # Write YAML subset
    config_yml_path = os.path.join(args.output_dir, "config.yml")
    with open(config_yml_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"vocab_size: {cfg['vocab_size']}\n")
        f_out.write(f"hidden_size: {cfg['hidden_size']}\n")
        f_out.write(f"embedding_size: {cfg.get('embedding_size', cfg['hidden_size'])}\n")
        f_out.write(f"num_hidden_layers: {cfg['num_hidden_layers']}\n")
        f_out.write(f"num_attention_heads: {cfg['num_attention_heads']}\n")
    # Copy token maps
    shutil.copy(args.token_maps, os.path.join(args.output_dir, "token_maps.pkl"))
    # Generate util.py with helper to load token map and wrapper for PL-BERT
    util_path = os.path.join(args.output_dir, "util.py")
    with open(util_path, "w", encoding="utf-8") as f:
        f.write('''"""
Utility functions for Somali PL-BERT integration with StyleTTS2.

This module exposes `load_token_map` to load the token dictionary and
`map_tokens` to convert phoneme strings into integer sequences.  It also
provides a `get_config` helper to return the model hyperparameters as a
dictionary.  Adjust these functions as needed to align with the StyleTTS2
frontend.
"""''')
        f.write("\n\nimport pickle\n\n")
        f.write("def load_token_map(path: str):\n")
        f.write("    with open(path, 'rb') as f:\n        return pickle.load(f)\n\n")
        f.write("def map_tokens(text: str, token_map: dict):\n")
        f.write("    return [token_map.get(tok, token_map.get('<unk>')) for tok in text.split()]\n\n")
        f.write("def get_config():\n")
        f.write(f"    return {{'vocab_size': {cfg['vocab_size']}, 'hidden_size': {cfg['hidden_size']}}}\n")
    logging.info("Packaged PL‑BERT into %s", args.output_dir)


if __name__ == "__main__":
    main()