#!/usr/bin/env python
"""
Command‑line tool to synthesise Somali speech using StyleTTS2 and a trained
Somali PL‑BERT encoder.  The script expects that StyleTTS2 is installed
and that a StyleTTS2 checkpoint has been trained with the provided
configuration files.  It loads the PL‑BERT package from `--plbert_dir`,
initialises the StyleTTS2 model, phonemizes the input text using the
fallback phonemizer (matching the training pipeline), and writes WAV
files to the output directory.

Example:

    python tts_infer_so.py \
      --text "Salaan! Sidee tahay?" \
      --plbert_dir runs/plbert_so/packaged \
      --styletts2_checkpoint path/to/styletts2.ckpt \
      --out out/

If `--text_file` is provided instead of `--text`, the script will read
multiple sentences (one per line) and synthesise each to a separate WAV.

Note: This script assumes that you have installed the `styletts2` Python
package and trained the acoustic models.  See the docs for instructions.
"""
import argparse
import json
import logging
import os
import uuid
import soundfile as sf

from phonemize.phonemizer_somali import phonemize_sentence


def load_plbert(plbert_dir: str):
    """Placeholder for loading PL‑BERT.  In practice, StyleTTS2's code will
    load the model from this directory.  Here we simply return the path."""
    return plbert_dir


def load_styletts2(checkpoint_path: str, plbert_dir: str):
    """Placeholder for loading a StyleTTS2 model.  Replace this with the
    appropriate import and initialisation code once StyleTTS2 is installed.
    """
    try:
        import styletts2
    except ImportError as e:
        raise ImportError("StyleTTS2 is not installed. Please install it from the official repo.") from e
    # Example API; adjust to match actual StyleTTS2 implementation
    model = styletts2.load_model(checkpoint_path, plbert_dir=plbert_dir)
    return model


def synthesize_sentences(sentences, model, plbert_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for idx, sentence in enumerate(sentences):
        # Phonemize sentence; returns phoneme string and grapheme string (we use phonemes)
        phonemes, _ = phonemize_sentence(sentence)
        # Convert phoneme string into integer IDs using token map
        # util.py in PL‑BERT package provides helper functions
        import importlib.util
        util_spec = importlib.util.spec_from_file_location("plbert_util", os.path.join(plbert_dir, "util.py"))
        util = importlib.util.module_from_spec(util_spec)
        util_spec.loader.exec_module(util)
        token_map = util.load_token_map(os.path.join(plbert_dir, "token_maps.pkl"))
        ids = util.map_tokens(phonemes, token_map)
        # Generate speech
        wav = model.tts(ids)
        # Save to file
        out_path = os.path.join(out_dir, f"sample_{idx:03d}.wav")
        sf.write(out_path, wav, 22050)
        logging.info("Synthesised %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Synthesize Somali speech using StyleTTS2 and PL‑BERT.")
    parser.add_argument("--text", type=str, help="Input text (Somali)")
    parser.add_argument("--text_file", type=str, help="Path to a file with one sentence per line")
    parser.add_argument("--plbert_dir", type=str, required=True, help="Path to packaged PL‑BERT directory")
    parser.add_argument("--styletts2_checkpoint", type=str, required=True, help="Trained StyleTTS2 checkpoint file")
    parser.add_argument("--out", type=str, required=True, help="Output directory for WAV files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if not (args.text or args.text_file):
        parser.error("Please provide --text or --text_file")
    # Load model
    plbert = load_plbert(args.plbert_dir)
    model = load_styletts2(args.styletts2_checkpoint, plbert)
    # Collect sentences
    sentences = []
    if args.text:
        sentences.append(args.text)
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            sentences.extend([line.strip() for line in f if line.strip()])
    synthesize_sentences(sentences, model, args.plbert_dir, args.out)


if __name__ == "__main__":
    main()