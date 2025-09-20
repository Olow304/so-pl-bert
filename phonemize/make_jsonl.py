#!/usr/bin/env python
"""
Split a JSONL dataset into training and development subsets.  Reads the
input JSONL file, shuffles the entries deterministically (using a fixed
random seed) and writes `train.jsonl` and `dev.jsonl` to the output
directory.  The proportion of data allocated to training can be controlled
with `--train_ratio` (default 0.95).

Example:

    python make_jsonl.py --input data_plbert/all.jsonl --train_ratio 0.95 --output data_plbert

"""
import argparse
import json
import logging
import os
import random

def main():
    parser = argparse.ArgumentParser(description="Split JSONL into train/dev.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file.")
    parser.add_argument("--train_ratio", type=float, default=0.95, help="Proportion of data for training.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for split files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    with open(args.input, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    random.Random(args.seed).shuffle(data)
    train_cutoff = int(len(data) * args.train_ratio)
    train_data = data[:train_cutoff]
    dev_data = data[train_cutoff:]
    os.makedirs(args.output, exist_ok=True)
    train_path = os.path.join(args.output, "train.jsonl")
    dev_path = os.path.join(args.output, "dev.jsonl")
    with open(train_path, "w", encoding="utf-8") as f_train:
        for item in train_data:
            f_train.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(dev_path, "w", encoding="utf-8") as f_dev:
        for item in dev_data:
            f_dev.write(json.dumps(item, ensure_ascii=False) + "\n")
    logging.info("Wrote %d training and %d dev examples to %s", len(train_data), len(dev_data), args.output)

if __name__ == "__main__":
    main()