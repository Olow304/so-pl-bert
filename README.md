# Somali PL‑BERT + StyleTTS2 Pipeline

This repository contains a full end‑to‑end pipeline for collecting large‑scale Somali text data, preparing it for a phoneme‑level BERT (PL‑BERT), training a Somali PL‑BERT model either from scratch or by continuing pre‑training from the official multilingual checkpoint, and integrating the resulting encoder into the StyleTTS2 speech synthesis system.  The code is organized so that every step—from crawling and cleaning raw text all the way to synthesising audio—can be reproduced with a single `make` command on a machine equipped with a single NVIDIA A100 GPU.

## Quick start

Clone this repository and run the following commands from the repository root:

```bash
# create the Python environment (conda or virtualenv) and install pinned dependencies
make env

# collect and clean Somali text, run language detection and deduplication, then
# phonemize and build token maps for PL‑BERT
make data

# continue pre‑training from the multilingual PL‑BERT model on Somali data
make cpt

# (alternatively) train a Somali PL‑BERT model from scratch
# make train

# package the trained model and token maps into a folder ready for StyleTTS2
make pack

# run a quick smoke test to synthesize a few Somali sentences using StyleTTS2
python styletts2_integration/tts_infer_so.py \
  --text "Salaan! Sidee tahay?" \
  --plbert_dir runs/plbert_so/packaged \
  --out out/

# listen to the resulting WAV files in the `out/` directory.
```

All scripts take a number of command‑line options (for example to change the number of GPU training steps, adjust batch size, or override input/output paths).  See the docstrings inside each script or run them with the `-h` flag for details.

## Project structure

The repository is organised into several top‑level directories:

| Path | Purpose |
|---|---|
| `crawl/` | Web crawlers for Wikipedia, Common Crawl derivatives (OSCAR/CC100), OPUS corpora and Somali news sites.  These scripts respect `robots.txt` and record licences and sources. |
| `data_prep/` | Normalisation, language identification, deduplication and sentence splitting utilities. |
| `phonemize/` | Phonemization pipeline including an eSpeak NG check, a fallback Somali G2P, and tools to generate JSONL training data and token maps. |
| `training/` | Scripts for training a Somali PL‑BERT model from scratch or by continuing pre‑training from Papercup’s multilingual checkpoint.  Also contains evaluation utilities. |
| `styletts2_integration/` | YAML configuration files for StyleTTS2, a command‑line inference driver, and a quick smoke test. |
| `docs/` | Documentation of the phoneset, pipeline overview, evaluation methodology and a licensing log detailing where data was obtained and under what terms. |

For the detailed design rationale, please refer to the files in `docs/`.  In particular, `phoneset.md` explains the Somali phoneset (with vowel length, gemination, uvular and pharyngeal consonants) and how it is normalised; `pipeline.md` diagrams the data flow end‑to‑end; `evaluation.md` lists both intrinsic and subjective evaluation procedures; and `licensing_log.md` records the licences and usage restrictions for each data source.
