# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Somali PL-BERT + StyleTTS2 pipeline for training phoneme-level BERT models and integrating them with StyleTTS2 for Somali speech synthesis. The pipeline handles everything from data collection to audio synthesis.

## Key Commands

### Complete Pipeline (PL-BERT + StyleTTS2)

```bash
# 1. Train PL-BERT (already done)
make data          # Prepare text data
make train         # Train PL-BERT
make pack          # Package for StyleTTS2

# 2. Prepare audio data for StyleTTS2
python prepare_styletts2_data.py  # Download & format Somali TTS dataset

# 3. Fine-tune StyleTTS2
python finetune_styletts2_somali.py  # Setup and train StyleTTS2

# 4. Generate speech
python styletts2_integration/tts_infer_so.py \
  --text "Your Somali text" \
  --plbert_dir runs/plbert_so/packaged \
  --styletts2_checkpoint Models/Somali/best_model.pth \
  --out output/
```

### Environment Setup
```bash
make env           # Create conda/virtualenv with pinned dependencies
```

### Data Pipeline
```bash
make data          # Complete data pipeline: crawl, clean, deduplicate, phonemize
```

### Training
```bash
make train         # Train Somali PL-BERT from scratch
make cpt           # Continue pretraining from multilingual PL-BERT (recommended)
```

### Model Packaging & Evaluation
```bash
make eval          # Run intrinsic evaluations on dev set
make pack          # Package trained model for StyleTTS2
```

### TTS Inference
```bash
python styletts2_integration/tts_infer_so.py \
  --text "Your Somali text" \
  --plbert_dir runs/plbert_so/packaged \
  --out out/
```

## Architecture

### Data Flow
1. **Raw Data Collection** (`crawl/`): Wikipedia, OSCAR, OPUS corpora scrapers
2. **Data Processing** (`data_prep/`): Cleaning, normalization, language ID, deduplication, sentence splitting
3. **Phonemization** (`phonemize/`): Convert text to phonemes using eSpeak NG with Somali G2P fallback
4. **Model Training** (`training/`): PL-BERT training (from scratch or continued pretraining)
5. **TTS Integration** (`styletts2_integration/`): StyleTTS2 configs and inference

### Key Components

- **Token Maps**: Built from phonemized data, stored in `phonemize/token_maps.pkl`
- **Training Data**: JSONL format with phoneme/grapheme pairs in `data_plbert/`
- **Model Checkpoints**: Saved in `runs/plbert_so/`
- **Packaged Models**: Ready for StyleTTS2 in `runs/plbert_so/packaged/`

### Somali Language Specifics

The system handles Somali-specific phonetic features:
- Vowel length distinctions (short vs long)
- Consonant gemination (doubling)
- Uvular consonants (q)
- Pharyngeal consonants (c, x)

See `docs/phoneset.md` for detailed phoneme inventory and normalization rules.

## Training Configurations

The pipeline supports two training approaches:
1. **From Scratch**: Uses `training/train_plbert_so.py` with fresh initialization
2. **Continue Pretraining**: Uses `training/continue_pretrain_plbert_so.py` starting from `papercup-ai/multilingual-pl-bert`

Default training uses single A100 GPU. Adjust batch size and training steps via command-line arguments in the respective scripts.