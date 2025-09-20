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
from transformers import AlbertForMaskedLM, Trainer, TrainingArguments


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

    # Custom data collator for masked LM (same as in training script)
    class CustomMLMDataCollator:
        def __init__(self, token_to_id, mlm_probability=0.15, max_length=256):
            self.token_to_id = token_to_id
            self.mlm_probability = mlm_probability
            self.max_length = max_length
            self.pad_token_id = token_to_id["<pad>"]
            self.mask_token_id = token_to_id["<mask>"]

        def __call__(self, examples):
            # Pad sequences
            padded_inputs = []
            attention_masks = []

            for example in examples:
                input_ids = example["input_ids"][:self.max_length]
                padding_length = self.max_length - len(input_ids)

                padded_input = input_ids + [self.pad_token_id] * padding_length
                attention_mask = [1] * len(input_ids) + [0] * padding_length

                padded_inputs.append(padded_input)
                attention_masks.append(attention_mask)

            # Convert to tensors
            input_ids = torch.tensor(padded_inputs, dtype=torch.long)
            attention_mask = torch.tensor(attention_masks, dtype=torch.long)

            # Create labels (copy of input_ids)
            labels = input_ids.clone()

            # Apply masking
            probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
            # Don't mask padding tokens
            probability_matrix.masked_fill_(input_ids == self.pad_token_id, value=0.0)

            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # Only compute loss on masked tokens

            # Replace masked positions with mask token
            input_ids[masked_indices] = self.mask_token_id

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    data_collator = CustomMLMDataCollator(
        token_to_id=token_to_id,
        mlm_probability=0.15,
        max_length=256
    )

    # Trainer in eval mode
    training_args = TrainingArguments(
        output_dir="/tmp/eval",
        per_device_eval_batch_size=args.batch_size,
        fp16=False,
        no_cuda=True,  # Use CPU for eval
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
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