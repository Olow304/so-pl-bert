#!/usr/bin/env python
"""
Deduplicate sentences using a simple hash set.  Optionally uses MinHash
similarity from the `datasketch` library to filter near duplicates.  This
module reads text files from the input directory and writes unique lines to
the output directory.  A summary report of the number of total vs unique
sentences is printed.

Example:

    python dedupe.py --input data_clean_filtered --output data_unique

"""
import argparse
import logging
import os
from datasketch import MinHash, MinHashLSH

def deduplicate_file(in_path: str, out_path: str, threshold: float = 0.8, use_minhash: bool = False):
    seen_hashes = set()
    if use_minhash:
        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        minhashes = {}
    total = 0
    unique = 0
    with open(in_path, encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            total += 1
            text = line.strip()
            if not text:
                continue
            if use_minhash:
                mh = MinHash(num_perm=128)
                for token in text.split():
                    mh.update(token.encode("utf-8"))
                # Check for near duplicate
                dup = False
                for h in lsh.query(mh):
                    dup = True
                    break
                if dup:
                    continue
                # Add new signature
                lsh.insert(len(minhashes), mh)
                minhashes[len(minhashes)] = text
                f_out.write(text + "\n")
                unique += 1
            else:
                h = hash(text)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    f_out.write(text + "\n")
                    unique += 1
    logging.info("Deduplication %s: total %d → unique %d", in_path, total, unique)

def main():
    parser = argparse.ArgumentParser(description="Deduplicate Somali sentences.")
    parser.add_argument("--input", type=str, required=True, help="Directory containing language‑filtered text files.")
    parser.add_argument("--output", type=str, required=True, help="Directory to write deduplicated files.")
    parser.add_argument("--use_minhash", action="store_true", help="Use MinHash LSH for near‑duplicate removal.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output, exist_ok=True)
    for filename in os.listdir(args.input):
        if not filename.endswith(".txt"):
            continue
        in_path = os.path.join(args.input, filename)
        out_path = os.path.join(args.output, filename)
        deduplicate_file(in_path, out_path, use_minhash=args.use_minhash)

if __name__ == "__main__":
    main()