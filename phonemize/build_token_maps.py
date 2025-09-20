#!/usr/bin/env python
"""
Build token maps for PL‑BERT.  Reads a JSONL file containing `phonemes`
and `graphemes` fields and constructs a vocabulary mapping tokens to
integers.  The union of phoneme and grapheme tokens is used as the
vocabulary.  Special tokens `<pad>`, `<mask>` and `<unk>` are added to the
start of the vocabulary.  The resulting mapping is pickled to the output
path.

This mapping is used by the PL‑BERT model to convert token sequences into
integers.  The `<mask>` token is used during MLM pre‑training and the
`<unk>` token handles any out‑of‑vocabulary tokens at inference time.

Example:

    python build_token_maps.py --input data_plbert/all.jsonl --output phonemize/token_maps.pkl

"""
import argparse
import json
import logging
import pickle


def main():
    parser = argparse.ArgumentParser(description="Build token map from phoneme/grapheme JSONL.")
    parser.add_argument("--input", type=str, required=True, help="Path to JSONL file with phonemes/graphemes.")
    parser.add_argument("--output", type=str, required=True, help="Path to save token_map.pkl.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    vocab = set()
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            vocab.update(obj["phonemes"].split())
            vocab.update(obj["graphemes"].split())
    # Remove potential empty token
    vocab.discard("")
    # Add special tokens at the beginning
    special_tokens = ["<pad>", "<mask>", "<unk>"]
    all_tokens = special_tokens + sorted(vocab)
    token_to_id = {tok: i for i, tok in enumerate(all_tokens)}
    with open(args.output, "wb") as f_out:
        pickle.dump(token_to_id, f_out)
    logging.info("Built vocabulary of size %d and saved to %s", len(token_to_id), args.output)


if __name__ == "__main__":
    main()