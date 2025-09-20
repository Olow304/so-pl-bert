# Somali PL‑BERT + StyleTTS2 Pipeline

This repository delivers a complete, reproducible implementation for building a Somali phoneme‑level BERT (PL‑BERT) and integrating it into the StyleTTS2 text‑to‑speech system.  It covers data collection and cleaning, phonemization, PL‑BERT training (from scratch or continued pre‑training), packaging for StyleTTS2 and inference.  The included Makefile orchestrates the workflow with a few commands.

## Repository structure

```
somali_plbert_styletts2/
├── Makefile                  # make targets for env, data, train, cpt, eval, pack
├── environment.yml           # conda environment definition with pinned versions
├── requirements.txt          # pip requirements for virtualenv installs
├── crawl/                    # web crawlers for Wikipedia and OSCAR/CC100
│   ├── crawl_wikipedia.py    # download Somali Wikipedia via HF dataset
│   └── fetch_oscar.py        # fetch OSCAR/CC100 Somali splits
├── data_prep/                # text normalisation, lang‑ID filtering, dedup & sentence splitting
│   ├── clean_normalize.py
│   ├── langid_filter.py
│   ├── dedupe.py
│   └── split_sentences.py
├── phonemize/                # phonemization pipeline
│   ├── check_espeak_so.py    # detect Somali voice in eSpeak NG【371023369836432†L88-L104】
│   ├── phonemizer_somali.py  # fallback rule‑based G2P covering long vowels and geminates【275631273323948†L200-L208】【275631273323948†L288-L292】
│   ├── phonemize_so.py       # use eSpeak NG or fallback to phonemize sentences
│   ├── build_token_maps.py   # construct vocabulary and token→ID map
│   └── make_jsonl.py         # shuffle and split dataset into train/dev
├── training/                 # PL‑BERT training, evaluation and packaging
│   ├── train_plbert_so.py    # train from scratch (MLM only)【657986872986870†L71-L80】
│   ├── continue_pretrain_plbert_so.py  # continue pre‑training from multilingual PL‑BERT
│   ├── eval_plbert_so.py     # compute MLM loss and perplexity
│   └── pack.py               # package model for StyleTTS2
├── styletts2_integration/
│   ├── train_first_somali.yaml   # Stage‑1 training config for StyleTTS2
│   ├── train_second_somali.yaml  # Stage‑2 training config with low LR for PL‑BERT【537144285036490†L1407-L1413】
│   ├── tts_infer_so.py           # CLI for inference using StyleTTS2 & PL‑BERT
│   └── quick_sanity_so.sh        # shell script to synthesise test sentences
├── docs/
│   ├── phoneset.md           # details of Somali phonology, vowel length & gemination【275631273323948†L200-L208】【275631273323948†L288-L292】
│   └── pipeline.md           # overview of the pipeline and evaluation methodology
└── README.md                # quick start instructions and high‑level overview
```

## Key design points

- **Data sources:** Somali Wikipedia (CC‑BY‑SA 3.0), OSCAR/CC100 derived from Common Crawl (CC0)【263184386926965†L96-L100】【601166331478072†L7-L13】, and other parallel corpora or news sites can be added under `crawl/` with licence compliance.
- **Phonology support:** The fallback phonemizer encodes the five long/short vowel pairs and geminate consonants【275631273323948†L200-L208】【275631273323948†L288-L292】.  Pharyngeal consonants (`ʕ`, `ħ`) and the uvular stop (`q`) are explicitly mapped.
- **eSpeak integration:** If `espeak-ng` is installed with a Somali voice, the scripts use `espeak-ng -v so -x` to generate phoneme mnemonics【371023369836432†L88-L104】.  Otherwise, a rule‑based phonemizer approximates Somali G2P.
- **PL‑BERT objectives:** The scripts focus on masked language modelling (MLM).  The original PL‑BERT also predicts graphemes from masked phonemes【657986872986870†L71-L80】; implementing this P2G task would require a custom model head.
- **StyleTTS2 integration:** YAML configs specify low learning rates for fine‑tuning the PL‑BERT encoder during Stage‑2 training【537144285036490†L1407-L1413】.  The inference script shows how to load the packaged PL‑BERT and synthesise Somali speech.

## Usage summary

1. **Create environment:** `make env` sets up a conda environment with pinned dependencies (Python 3.11, PyTorch 2.1.2, Transformers 4.35.2, etc.).
2. **Prepare data:** `make data` downloads Somali corpora, cleans and deduplicates them, phonemizes sentences and builds vocabulary/token maps.  The processed data lives in `data_plbert/`.
3. **Train PL‑BERT:** Use `make cpt` to continue pre‑training from the multilingual PL‑BERT checkpoint, or `make train` to train from scratch.  Checkpoints are written to `runs/plbert_so/`.
4. **Evaluate:** `make eval` computes masked LM loss and perplexity on the dev set.
5. **Package:** `make pack` exports the trained model into a folder with `step_000001.pt`, `config.yml`, `token_maps.pkl` and `util.py`.  Copy this folder into `StyleTTS2/Utils/PLBERT_Somali/` for use in StyleTTS2.
6. **Synthesise:** Train StyleTTS2 using the provided YAML configs.  Then run `python styletts2_integration/tts_infer_so.py --text "Salaan!" --plbert_dir runs/plbert_so/packaged --styletts2_checkpoint <ckpt> --out out/` to generate Somali speech.

## Notes and caveats

- The crawling scripts currently demonstrate how to fetch Somali Wikipedia and OSCAR/CC100 splits via HuggingFace.  Additional crawlers should be added for news sites and OPUS corpora, respecting robots.txt and licences.
- FastText’s language identifier must be downloaded (`lid.176.bin`) for language filtering; if unavailable the script proceeds without filtering.
- The fallback phonemizer is intentionally simple; replacing it with a neural G2P model would improve phonetic accuracy.
- The evaluation and subjective listening test guidelines can be extended; the repository provides a template but further metrics (e.g., P2G accuracy, prosodic probing) may be added.

By following the steps above, one can build a Somali PL‑BERT and integrate it into StyleTTS2, achieving natural‑sounding speech with correct vowel length, gemination and pharyngeal consonants.