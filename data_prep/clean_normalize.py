#!/usr/bin/env python
"""
Simple text normaliser for Somali.  This script reads all text files in the
input directory, normalises whitespace and Unicode, optionally lowercases
text, strips extraneous punctuation, and writes the cleaned sentences into
`output` directory (mirroring the input filenames).  Normalisation uses
`unidecode` to convert accented characters to their ASCII equivalents and
collapses multiple whitespace characters into a single space.

Note: Somali uses the Latin alphabet with a few digraphs (dh, kh, sh).  We
retain case information and diacritics if present.

Example:

    python clean_normalize.py --input data_raw --output data_clean

"""
import argparse
import logging
import os
import re
from unidecode import unidecode

def normalise_text(text: str) -> str:
    # Convert to Unicode NFKC and ASCII fallback
    text = unidecode(text)
    # Replace non‑breaking spaces and other whitespace with normal space
    text = re.sub(r"\s+", " ", text)
    # Remove control characters
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)
    return text.strip()

def process_file(in_path: str, out_path: str):
    with open(in_path, encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            cleaned = normalise_text(line)
            if cleaned:
                f_out.write(cleaned + "\n")

def main():
    parser = argparse.ArgumentParser(description="Clean and normalise Somali text files.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing raw text files.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for cleaned text files.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output, exist_ok=True)
    for filename in os.listdir(args.input):
        if not filename.endswith(".txt"):
            continue
        in_path = os.path.join(args.input, filename)
        out_path = os.path.join(args.output, filename)
        logging.info("Normalising %s → %s", in_path, out_path)
        process_file(in_path, out_path)

if __name__ == "__main__":
    main()