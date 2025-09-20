#!/usr/bin/env python
"""
Evaluate a trained PL‑BERT model on the development set.  Computes the
average masked language model loss and reports the perplexity.  The script
loads the specified model checkpoint, the dev JSONL file and the
token_map.  P2G evaluation is not implemented in this simplified version.

Example:

    python training/eval_plbert_so.py \
      --model runs/plbert_so/continue \
      --dev data_plbert/dev.jsonl \
      --token_maps phonemize/token_maps.pkl

"""
import argparse
import json
import logging
import os
import pickle
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (AlbertForMaskedLM, Trainer,
                          DataCollatorForLanguageModeling)


def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def encode_data(data: List[Dict], token_to_id: Dict[str, int]) -> List[List[int]]:
    unk = token_to_id["<unk>"]
    encoded = []
    for item in data:
        ids = [token_to_id.get(tok, unk) for tok in item["phonemes"].split()]
        encoded.append(ids)
    return encoded


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PL‑BERT model.")
    parser.add_argument("--model", type=str, required=True, help="Directory containing the saved model.")
    parser.add_argument("--dev", type=str, required=True, help="Dev JSONL file.")
    parser.add_argument("--token_maps", type=str, required=True, help="Pickled token map.")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    # Load model
    model = AlbertForMaskedLM.from_pretrained(args.model)
    # Load token map
    with open(args.token_maps, "rb") as f:
        token_to_id = pickle.load(f)
    # Encode dev data
    dev_data = load_jsonl(args.dev)
    dev_enc = encode_data(dev_data, token_to_id)
    dev_ds = Dataset.from_dict({"input_ids": dev_enc})
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=None,
        mlm=True,
        mlm_probability=0.15,
        pad_token_id=token_to_id["<pad>"],
    )
    # Trainer in eval mode
    training_args = dict(
        output_dir="/tmp/eval", per_device_eval_batch_size=args.batch_size, fp16=torch.cuda.is_available()
    )
    trainer = Trainer(
        model=model,
        args=type("Args", (), training_args)(),
        eval_dataset=dev_ds,
        data_collator=data_collator,
    )
    eval_output = trainer.evaluate()
    mlm_loss = eval_output.get("eval_loss")
    perplexity = torch.exp(torch.tensor(mlm_loss)).item() if mlm_loss is not None else None
    logging.info("Eval loss: %.4f", mlm_loss)
    if perplexity is not None:
        logging.info("Perplexity: %.2f", perplexity)


if __name__ == "__main__":
    main()