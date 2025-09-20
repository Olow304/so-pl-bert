#!/usr/bin/env python
"""
Language identification filter.  Uses Facebook's fastText model to detect
language of each line and filters out sentences that are not predicted to be
Somali with high confidence.  The `lid.176.bin` model is loaded from
https://fasttext.cc/docs/en/language-identification.html.  Download it
before running this script if not already present.  If the model is
missing, the script will skip filtering.

Example:

    python langid_filter.py --input data_clean --output data_clean_filtered

"""
import argparse
import logging
import os
import fasttext

def load_model(model_path: str):
    try:
        return fasttext.load_model(model_path)
    except Exception as e:
        logging.warning("Failed to load fastText model (%s). Language filtering disabled.", e)
        return None

def filter_file(model, in_path: str, out_path: str, threshold: float = 0.8):
    with open(in_path, encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            text = line.strip()
            if not text:
                continue
            if model is None:
                f_out.write(text + "\n")
                continue
            label, prob = model.predict(text.replace("\n", " "))
            lang = label[0].replace("__label__", "")
            if lang == "so" and prob[0] >= threshold:
                f_out.write(text + "\n")

def main():
    parser = argparse.ArgumentParser(description="Filter Somali sentences using fastText language identification.")
    parser.add_argument("--input", type=str, required=True, help="Directory with cleaned text files.")
    parser.add_argument("--output", type=str, required=True, help="Directory to write filtered text files.")
    parser.add_argument("--model", type=str, default="lid.176.bin", help="Path to fastText language id model.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Probability threshold for Somali detection.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    model = load_model(args.model)
    os.makedirs(args.output, exist_ok=True)
    for filename in os.listdir(args.input):
        if not filename.endswith(".txt"):
            continue
        in_path = os.path.join(args.input, filename)
        out_path = os.path.join(args.output, filename)
        logging.info("Language filtering %s → %s", in_path, out_path)
        filter_file(model, in_path, out_path, threshold=args.threshold)

if __name__ == "__main__":
    main()