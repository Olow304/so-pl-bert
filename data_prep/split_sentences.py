#!/usr/bin/env python
"""
Split deduplicated documents into sentences.  Somali orthography uses
periods, question marks and exclamation marks to mark sentence boundaries.
This script uses a simple regular expression to segment each line into
sentences.  The resulting sentences are written to the output directory.

Example:

    python split_sentences.py --input data_unique --output data_sentences

"""
import argparse
import logging
import os
import re

def split_line_to_sentences(text: str) -> list[str]:
    # Split at ., ?, ! followed by whitespace or end of line.  Keep punctuation.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]

def process_file(in_path: str, out_path: str):
    with open(in_path, encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            sentences = split_line_to_sentences(line)
            for sent in sentences:
                f_out.write(sent + "\n")

def main():
    parser = argparse.ArgumentParser(description="Split Somali text into sentences.")
    parser.add_argument("--input", type=str, required=True, help="Directory with deduplicated text files.")
    parser.add_argument("--output", type=str, required=True, help="Directory to write sentences.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output, exist_ok=True)
    for filename in os.listdir(args.input):
        if not filename.endswith(".txt"):
            continue
        in_path = os.path.join(args.input, filename)
        out_path = os.path.join(args.output, filename)
        logging.info("Splitting sentences %s → %s", in_path, out_path)
        process_file(in_path, out_path)

if __name__ == "__main__":
    main()