#!/usr/bin/env python
"""
Crawl Somali Wikipedia articles using the HuggingFace `wikipedia` dataset.  The
dataset provides dumps of Wikipedia pages in many languages.  This script
extracts the text of each article in the Somali dump, strips any
section headings or markup, and writes one article per line to an
output file.  The script respects the Creative Commons Attribution
ShareAlike 3.0 licence of Wikipedia; see
https://dumps.wikimedia.org/legal.html for details.

Example:

    python crawl_wikipedia.py --out data_raw/wikipedia.txt

"""
import argparse
import logging
import os
import re
from datasets import load_dataset

def extract_plain_text(page: str) -> str:
    """Remove headings and extra whitespace from a Wikipedia article."""
    # Remove markup: headers (lines starting with =), templates etc.
    text = re.sub(r'=+[^=]+=+', ' ', page)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    parser = argparse.ArgumentParser(description="Download Somali Wikipedia dump and extract plain text.")
    parser.add_argument("--out", type=str, required=True, help="Path to write the extracted articles (one per line).")
    parser.add_argument("--skip-on-error", action="store_true", help="Skip if Wikipedia data unavailable")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    dataset = None
    try:
        logging.info("Loading Somali Wikipedia dataset via HuggingFace…")
        # Try different Wikipedia dataset formats
        for attempt in [
            lambda: load_dataset("wikimedia/wikipedia", "20231101.so", split="train"),
            lambda: load_dataset("wikipedia", "20220301.so", split="train"),
            lambda: load_dataset("wikipedia", "20220301.so", split="train", trust_remote_code=True),
        ]:
            try:
                dataset = attempt()
                break
            except Exception as e:
                logging.debug(f"Attempt failed: {e}")
                continue
    except Exception as e:
        logging.error(f"All attempts to load Wikipedia dataset failed: {e}")

    if dataset is None:
        logging.warning("Could not load Wikipedia dataset. Creating placeholder file.")
        with open(args.out, "w", encoding="utf-8") as f_out:
            f_out.write("# Wikipedia data unavailable - placeholder file\n")
        logging.info("Created placeholder file at %s", args.out)
        return

    count = 0
    with open(args.out, "w", encoding="utf-8") as f_out:
        for article in dataset:
            text = article.get("text", "")
            plain = extract_plain_text(text)
            if plain:
                f_out.write(plain + "\n")
                count += 1
    logging.info("Wrote %d Somali Wikipedia articles to %s", count, args.out)

if __name__ == "__main__":
    main()