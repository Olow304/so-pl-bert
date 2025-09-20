#!/usr/bin/env python
"""
Fetch the Somali subset of OSCAR or CC100 from the HuggingFace hub.  OSCAR is a
multilingual corpus built from Common Crawl using language classification and
filtering.  See the OSCAR dataset card for details and licensing
information (released under CC0 1.0)【263184386926965†L96-L100】.  This script
downloads the Somali portion and writes one document per line to an
output file.

Example:

    python fetch_oscar.py --out data_raw/oscar.txt

"""
import argparse
import logging
import os
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Download the Somali OSCAR or CC100 corpus from HuggingFace.")
    parser.add_argument("--out", type=str, required=True, help="Output text file (one document per line).")
    parser.add_argument("--dataset", type=str, default="oscar", choices=["oscar", "cc100"], help="Dataset to download: oscar or cc100")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    subset_name = "unshuffled_deduplicated_so" if args.dataset == "oscar" else "so"  # cc100 uses 'so'
    dataset_name = "oscar" if args.dataset == "oscar" else "cc100"
    logging.info("Loading %s subset %s…", dataset_name, subset_name)
    dataset = load_dataset(dataset_name, subset_name, split="train")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    count = 0
    with open(args.out, "w", encoding="utf-8") as f_out:
        for item in dataset:
            text = item.get("text", "")
            if text:
                # Replace newlines in the document to keep one document per line
                text = text.replace("\n", " ").strip()
                if text:
                    f_out.write(text + "\n")
                    count += 1
    logging.info("Wrote %d documents from %s to %s", count, dataset_name, args.out)

if __name__ == "__main__":
    main()