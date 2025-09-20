#!/usr/bin/env python
"""
Continue pretraining a multilingual PL‑BERT on Somali data.  This script
downloads the pre‑trained model from HuggingFace (default:
`papercup-ai/multilingual-pl-bert`) and fine‑tunes it on Somali phoneme
sequences using the MLM objective.  If the Somali token map contains
additional tokens not present in the original vocabulary, the embeddings are
resized accordingly and the new slots are randomly initialised.  As in
`train_plbert_so.py`, only the MLM objective is implemented here for
simplicity.

Example:

    python training/continue_pretrain_plbert_so.py \
      --train data_plbert/train.jsonl \
      --dev data_plbert/dev.jsonl \
      --token_maps phonemize/token_maps.pkl \
      --model_name papercup-ai/multilingual-pl-bert \
      --out_dir runs/plbert_so/continue

"""
import argparse
import json
import logging
import os
import pickle
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (AlbertForMaskedLM, Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling)


def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def encode_data(data: List[Dict], token_to_id: Dict[str, int]) -> List[List[int]]:
    encoded = []
    unk = token_to_id["<unk>"]
    for item in data:
        ids = [token_to_id.get(tok, unk) for tok in item["phonemes"].split()]
        encoded.append(ids)
    return encoded


def main():
    parser = argparse.ArgumentParser(description="Continue pretraining multilingual PL‑BERT on Somali.")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--token_maps", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="papercup-ai/multilingual-pl-bert")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Load token map
    with open(args.token_maps, "rb") as f:
        token_to_id = pickle.load(f)
    vocab_size = len(token_to_id)

    # Load pretrained model
    logging.info("Loading pretrained model %s", args.model_name)
    model = AlbertForMaskedLM.from_pretrained(args.model_name)
    # Resize embeddings if necessary
    current_vocab = model.config.vocab_size
    if vocab_size != current_vocab:
        logging.info("Resizing embeddings: pretrained vocab %d → new vocab %d", current_vocab, vocab_size)
        model.resize_token_embeddings(vocab_size)
    # Load data
    train_data = load_jsonl(args.train)
    dev_data = load_jsonl(args.dev)
    train_enc = encode_data(train_data, token_to_id)
    dev_enc = encode_data(dev_data, token_to_id)
    train_ds = Dataset.from_dict({"input_ids": train_enc})
    dev_ds = Dataset.from_dict({"input_ids": dev_enc})
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=None,
        mlm=True,
        mlm_probability=0.15,
        pad_token_id=token_to_id["<pad>"],
    )
    # Training arguments
    os.makedirs(args.out_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    logging.info("Continued pretraining complete. Model saved to %s", args.out_dir)


if __name__ == "__main__":
    main()