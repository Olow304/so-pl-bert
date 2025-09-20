#!/usr/bin/env python
"""
Train a Somali PL‑BERT model from scratch.  This script uses the HuggingFace
Transformers library to instantiate an ALBERT configuration with a
vocabulary based on the generated token map and trains it on masked
language modelling (MLM).  Although the original PL‑BERT objective includes
grapheme prediction (P2G)【657986872986870†L71-L80】, this simplified trainer focuses
on the phoneme MLM component for clarity.  Implementing P2G would require
customising the model head to jointly predict grapheme tokens for the
masked positions, which can be added later.

Example usage:

    python training/train_plbert_so.py \
      --train data_plbert/train.jsonl \
      --dev data_plbert/dev.jsonl \
      --token_maps phonemize/token_maps.pkl \
      --out_dir runs/plbert_so/from_scratch

"""
import argparse
import json
import logging
import os
import pickle
from dataclasses import dataclass
from typing import List, Dict

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (AlbertConfig, AlbertForMaskedLM, Trainer,
                          TrainingArguments, DataCollatorForLanguageModeling)


def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def encode_data(data: List[Dict], token_to_id: Dict[str, int]) -> List[List[int]]:
    """Convert phoneme token strings to lists of integer IDs."""
    encoded = []
    for item in data:
        tokens = item["phonemes"].split()
        ids = [token_to_id.get(tok, token_to_id["<unk>"]) for tok in tokens]
        encoded.append(ids)
    return encoded


def main():
    parser = argparse.ArgumentParser(description="Train PL‑BERT from scratch (phoneme MLM only).")
    parser.add_argument("--train", type=str, required=True, help="Path to training JSONL file.")
    parser.add_argument("--dev", type=str, required=True, help="Path to dev JSONL file.")
    parser.add_argument("--token_maps", type=str, required=True, help="Pickled token_to_id mapping.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for checkpoints.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per device.")
    parser.add_argument("--max_len", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Load vocabulary
    with open(args.token_maps, "rb") as f:
        token_to_id = pickle.load(f)
    vocab_size = len(token_to_id)
    logging.info("Loaded vocabulary of size %d", vocab_size)

    # Load data
    train_data = load_jsonl(args.train)
    dev_data = load_jsonl(args.dev)
    train_enc = encode_data(train_data, token_to_id)
    dev_enc = encode_data(dev_data, token_to_id)
    logging.info("Loaded %d train and %d dev examples", len(train_enc), len(dev_enc))

    # Wrap in HuggingFace Dataset
    train_ds = Dataset.from_dict({"input_ids": train_enc})
    dev_ds = Dataset.from_dict({"input_ids": dev_enc})

    # Define config
    config = AlbertConfig(
        vocab_size=vocab_size,
        embedding_size=128,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=args.max_len,
        type_vocab_size=1,
        pad_token_id=token_to_id["<pad>"],
        bos_token_id=None,
        eos_token_id=None,
    )
    model = AlbertForMaskedLM(config)

    # Data collator for masked LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=None,  # we provide input_ids directly
        mlm=True,
        mlm_probability=0.15,
        pad_token_id=token_to_id["<pad>"],
    )

    # Prepare training arguments
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

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    logging.info("Training complete. Model saved to %s", args.out_dir)


if __name__ == "__main__":
    main()