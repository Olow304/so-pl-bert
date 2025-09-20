# Pipeline Overview

This document describes the end‑to‑end pipeline for constructing a Somali
phoneme‑level BERT (PL‑BERT) and integrating it into the StyleTTS2 speech
synthesis system.  The pipeline is divided into four main stages: data
collection and cleaning, phonemization and dataset preparation, PL‑BERT
training, and StyleTTS2 integration.  Each stage is backed by scripts
provided in this repository.

## 1. Data collection and cleaning

We aim to assemble a corpus of tens of millions of Somali tokens from
sources with permissive licences.  The following datasets and resources are
used:

- **Somali Wikipedia:** downloaded via the HuggingFace `wikipedia` dataset.
  Wikipedia content is licensed under CC BY‑SA 3.0; we respect this by
  retaining attribution and documenting usage.
- **OSCAR/CC100 Somali:** large corpora extracted from Common Crawl using the
  OSCAR pipeline【263184386926965†L96-L100】 and CCNet【601166331478072†L7-L13】.  OSCAR provides
  deduplicated text in dozens of languages; we fetch the Somali subset.
- **OPUS corpora:** Somali sides of bitext corpora such as GlobalVoices and
  JW300 (retrieval scripts can be added).  Only the Somali monolingual
  text is retained.
- **News sites:** VOA Somali, BBC Somali, Hiiraan Online, Garowe Online,
  Radio Muqdisho and others are crawled with domain‑specific scrapers and
  subject to robots.txt and licence checks.  These scripts can be added
  under `crawl/`.

All raw text is written to `data_raw/`.  Subsequent scripts perform the
following processing steps:

1. **Normalisation (`data_prep/clean_normalize.py`)**
   - Lowercase the text, normalise Unicode and collapse whitespace.
   - Remove control characters and extraneous punctuation.

2. **Language filtering (`data_prep/langid_filter.py`)**
   - Use Facebook’s fastText language‑ID model to retain only lines
     predicted to be Somali with high confidence.
   - If the model cannot be loaded, filtering is skipped.

3. **Deduplication (`data_prep/dedupe.py`)**
   - Simple hash‑based deduplication removes exact duplicate sentences.
   - Optional MinHash LSH can identify near duplicates; enable with
     `--use_minhash`.

4. **Sentence splitting (`data_prep/split_sentences.py`)**
   - Split each document into sentences at `.`, `?` and `!` boundaries.

After these steps, the corpus resides in `data_sentences/` with one sentence per line.

## 2. Phonemization and dataset preparation

The cleaned sentences are converted into phoneme/grapheme pairs using
`phonemize/phonemize_so.py`.  This script attempts to use **eSpeak NG** with
the Somali voice (`so`) and mnemonic phoneme mode (`-x`).  If eSpeak
returns an error or the Somali voice is missing, a fallback rule‑based G2P
defined in `phonemizer_somali.py` is used【371023369836432†L88-L104】.  Both methods
generate two parallel sequences:

- **Phonemes:** space‑separated phoneme tokens, with long vowels and
  digraphs treated as single tokens.
- **Graphemes:** space‑separated character (or digraph) tokens, mirroring
  the phoneme sequence with an underscore `_` marking word boundaries.

The script writes a single JSONL file with entries of the form
`{"phonemes": "...", "graphemes": "...", "source": "filename"}`.

`phonemize/build_token_maps.py` reads the full JSONL and constructs a
vocabulary of all observed phoneme and grapheme tokens.  Special tokens
`<pad>`, `<mask>` and `<unk>` are prepended.  The mapping from tokens to
integer IDs is pickled to `phonemize/token_maps.pkl`.  Finally
`phonemize/make_jsonl.py` shuffles the dataset with a fixed seed and splits
it into `train.jsonl` and `dev.jsonl` based on the desired training ratio.

## 3. Training PL‑BERT

Two training scripts are provided under `training/`:

1. **`train_plbert_so.py`** builds an ALBERT configuration from scratch
   based on the vocabulary size.  It trains a masked language model on
   phoneme sequences using the MLM objective.  The original PL‑BERT
   incorporates a phoneme‑to‑grapheme prediction task in addition to MLM【657986872986870†L71-L80】;
   implementing P2G requires a custom model head and is left as future
   work.  Training parameters such as learning rate, batch size and
   sequence length can be adjusted via command‑line options.

2. **`continue_pretrain_plbert_so.py`** downloads the multilingual PL‑BERT
   checkpoint from the HuggingFace hub and continues pretraining on
   Somali data.  If the Somali vocabulary introduces new tokens, the
   model’s embedding matrix is resized to accommodate them.  A low
   learning rate (e.g. `1e-5`) is used to fine‑tune the model while
   preserving its multilingual knowledge.

During training, intermediate checkpoints and logs are written to
`runs/plbert_so/...`.  Evaluation on the dev set is performed at the end of
each epoch.  `training/eval_plbert_so.py` can compute the masked LM loss
and perplexity on the held‑out set.

The packaging script `training/pack.py` collects the best checkpoint,
converts the configuration into a YAML file expected by StyleTTS2, copies
the token map and generates a small `util.py` helper.  The resulting
folder (e.g. `runs/plbert_so/packaged`) contains:

- `step_000001.pt` – PL‑BERT weights;
- `config.yml` – model hyperparameters;
- `token_maps.pkl` – token dictionary;
- `util.py` – helper functions for token lookup.

## 4. StyleTTS2 integration

The `styletts2_integration/` directory includes YAML templates for the
two‑stage training process of StyleTTS2 on Somali.  Stage 1 learns the
acoustic and textual representations from scratch, and stage 2 fine‑tunes
with more data and a lower learning rate for PL‑BERT.  Important hyper‑
parameters such as `plbert_lr` (learning rate for the PL‑BERT encoder)
should be kept small (e.g. `1e-5`)【537144285036490†L1407-L1413】.

The `tts_infer_so.py` script illustrates how to load the packaged PL‑BERT
and a trained StyleTTS2 checkpoint, phonemize Somali text and synthesise
speech.  The script uses the fallback phonemizer for phoneme sequences and
delegates audio generation to the StyleTTS2 model.  A shell script
`quick_sanity_so.sh` synthesises a few sample sentences to verify that the
pipeline is working end‑to‑end.

## Evaluation

Intrinsic evaluation of PL‑BERT is performed via masked LM loss and
perplexity on the dev set (`training/eval_plbert_so.py`).  To assess
phoneme‑to‑grapheme accuracy or prosodic competence, one can construct a
probe set of challenging Somali sentences (including long vowels,
geminates, pharyngeals, numerals and code‑switches) and measure the
model’s predictions.  Subjective evaluation involves listening tests
focusing on naturalness, rhythm, vowel length and gemination fidelity; a
MOS‑style questionnaire can be used.  Refer to `docs/evaluation.md` for
guidelines on designing these tests.