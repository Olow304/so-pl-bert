##
# Makefile for the Somali PL‑BERT and StyleTTS2 pipeline.
#
# Each target creates a reproducible environment, prepares data, trains the
# PL‑BERT model and packages the artefacts for consumption by StyleTTS2.
# Adjust the variables at the top of this file to suit your local setup.

PYTHON     ?= python
ENV_NAME   ?= plbert_so_env
CONDA      ?= $(shell command -v conda 2>/dev/null)

all: help

.PHONY: help
help:
	@echo "Targets:"
	@echo "  make env    - create conda/virtualenv with pinned dependencies"
	@echo "  make data   - crawl & clean Somali text, phonemize and build token maps"
	@echo "  make train  - train Somali PL‑BERT from scratch"
	@echo "  make cpt    - continue pretraining from multilingual PL‑BERT"
	@echo "  make eval   - run intrinsic evaluations on dev set"
	@echo "  make pack   - package trained model for StyleTTS2"

# Create the Python environment using conda if available, otherwise fallback to venv.
.PHONY: env
env:
	@if [ -n "$(CONDA)" ]; then \
		echo "[INFO] Creating conda environment $(ENV_NAME)"; \
		$(CONDA) env create -f environment.yml -n $(ENV_NAME); \
		echo "[INFO] To activate: conda activate $(ENV_NAME)"; \
	else \
		echo "[INFO] Creating virtualenv"; \
		$(PYTHON) -m venv $(ENV_NAME); \
		. $(ENV_NAME)/bin/activate && pip install -r requirements.txt; \
		echo "[INFO] Environment created. To activate: source $(ENV_NAME)/bin/activate"; \
	fi

# Run the complete data pipeline: crawl raw text, clean and dedup, phonemize
# into JSONL and create token maps.  The outputs are written to `data_plbert`.
.PHONY: data
data:
	-$(PYTHON) crawl/crawl_wikipedia.py --out data_raw/wikipedia.txt --skip-on-error
	-$(PYTHON) crawl/fetch_oscar.py --out data_raw/oscar.txt
	$(PYTHON) data_prep/clean_normalize.py --input data_raw --output data_clean
	$(PYTHON) data_prep/langid_filter.py --input data_clean --output data_clean_filtered
	$(PYTHON) data_prep/dedupe.py --input data_clean_filtered --output data_unique
	$(PYTHON) data_prep/split_sentences.py --input data_unique --output data_sentences
	# Convert sentences to phoneme/grapheme pairs
	$(PYTHON) phonemize/phonemize_so.py --input data_sentences --output data_plbert/all.jsonl
	# Build token maps from the full dataset
	$(PYTHON) phonemize/build_token_maps.py --input data_plbert/all.jsonl --output phonemize/token_maps.pkl
	# Split JSONL into train/dev sets
	$(PYTHON) phonemize/make_jsonl.py --input data_plbert/all.jsonl --train_ratio 0.95 --output data_plbert

# Train PL‑BERT from scratch
.PHONY: train
train:
	$(PYTHON) training/train_plbert_so.py \
		--train data_plbert/train.jsonl \
		--dev data_plbert/dev.jsonl \
		--token_maps phonemize/token_maps.pkl \
		--out_dir runs/plbert_so/from_scratch

# Continue pretraining from Papercup’s multilingual PL‑BERT
.PHONY: cpt
cpt:
	$(PYTHON) training/continue_pretrain_plbert_so.py \
		--train data_plbert/train.jsonl \
		--dev data_plbert/dev.jsonl \
		--token_maps phonemize/token_maps.pkl \
		--model_name papercup-ai/multilingual-pl-bert \
		--out_dir runs/plbert_so/continue

# Run intrinsic evaluation (masked LM loss & P2G accuracy)
.PHONY: eval
eval:
	$(PYTHON) training/eval_plbert_so.py \
		--model runs/plbert_so/continue/best_model.pt \
		--dev data_plbert/dev.jsonl \
		--token_maps phonemize/token_maps.pkl

# Package PL‑BERT for StyleTTS2
.PHONY: pack
pack:
	$(PYTHON) training/pack.py \
		--input_dir runs/plbert_so/continue \
		--token_maps phonemize/token_maps.pkl \
		--output_dir runs/plbert_so/packaged