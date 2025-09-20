#!/usr/bin/env python
"""
Prepare Somali TTS dataset for StyleTTS2 fine-tuning.
Downloads the Somalitts/jelle8000 dataset from HuggingFace and converts it to StyleTTS2 format.
"""
import os
import argparse
import logging
import json
import soundfile as sf
import librosa
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import sys
sys.path.append('phonemize')
from phonemizer_somali import phonemize_sentence

def prepare_styletts2_dataset(output_dir="data_styletts2", sample_rate=24000, max_duration=10.0):
    """
    Prepare the Somali TTS dataset for StyleTTS2 training.

    Args:
        output_dir: Directory to save prepared data
        sample_rate: Target sample rate (StyleTTS2 uses 24kHz)
        max_duration: Maximum audio duration in seconds
    """
    logging.basicConfig(level=logging.INFO)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    wavs_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    logging.info("Loading Somalitts/jelle8000 dataset from HuggingFace...")

    try:
        # Load the dataset
        dataset = load_dataset("Somalitts/jelle8000", split="train")
        logging.info(f"Loaded {len(dataset)} samples")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        logging.info("You may need to authenticate with HuggingFace:")
        logging.info("  huggingface-cli login")
        return

    # Prepare file lists
    train_list = []
    val_list = []
    ood_list = []  # Out-of-distribution for zero-shot

    # Process each sample
    total_duration = 0
    skipped_long = 0
    skipped_error = 0

    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        try:
            # Get audio and text
            audio_array = sample['audio']['array']
            orig_sr = sample['audio']['sampling_rate']
            text = sample['text'].strip()

            # Skip empty text
            if not text:
                continue

            # Resample to 24kHz if needed
            if orig_sr != sample_rate:
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=orig_sr,
                    target_sr=sample_rate
                )

            # Check duration
            duration = len(audio_array) / sample_rate
            if duration > max_duration:
                skipped_long += 1
                continue

            total_duration += duration

            # Save audio file
            audio_filename = f"som_{idx:06d}.wav"
            audio_path = os.path.join(wavs_dir, audio_filename)
            sf.write(audio_path, audio_array, sample_rate)

            # Phonemize the text
            phonemes, _ = phonemize_sentence(text)

            # Create entry for file list
            # Format: audiofile|phonemized_text|speaker
            # Using single speaker "somali" for now
            entry = f"{audio_filename}|{phonemes}|somali"

            # Split into train/val/OOD (80/10/10)
            if idx % 10 < 8:
                train_list.append(entry)
            elif idx % 10 == 8:
                val_list.append(entry)
            else:
                ood_list.append(entry)

        except Exception as e:
            logging.warning(f"Error processing sample {idx}: {e}")
            skipped_error += 1
            continue

    # Write file lists
    train_file = os.path.join(output_dir, "train_list.txt")
    val_file = os.path.join(output_dir, "val_list.txt")
    ood_file = os.path.join(output_dir, "OOD_list.txt")

    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_list))

    with open(val_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_list))

    with open(ood_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ood_list))

    # Print statistics
    logging.info("=" * 70)
    logging.info("Dataset preparation complete!")
    logging.info(f"Total samples processed: {len(train_list) + len(val_list) + len(ood_list)}")
    logging.info(f"Train samples: {len(train_list)}")
    logging.info(f"Validation samples: {len(val_list)}")
    logging.info(f"OOD samples: {len(ood_list)}")
    logging.info(f"Total duration: {total_duration/3600:.2f} hours")
    logging.info(f"Skipped (too long): {skipped_long}")
    logging.info(f"Skipped (errors): {skipped_error}")
    logging.info("=" * 70)
    logging.info(f"Files saved to: {output_dir}")
    logging.info(f"  - {train_file}")
    logging.info(f"  - {val_file}")
    logging.info(f"  - {ood_file}")
    logging.info(f"  - Audio files in: {wavs_dir}")

    # Create a sample config for StyleTTS2
    create_styletts2_config(output_dir)

def create_styletts2_config(data_dir):
    """Create a configuration file for StyleTTS2 fine-tuning."""

    config = {
        "log_dir": "Models/Somali",
        "save_freq": 1,
        "log_interval": 10,
        "device": "cuda",
        "epochs": 50,  # Adjust based on data amount
        "batch_size": 4,  # Adjust based on GPU memory
        "max_len": 400,  # ~5 seconds at 24kHz
        "pretrained_model": "Models/LibriTTS/epochs_2nd_00020.pth",  # Pre-trained checkpoint from HuggingFace
        "second_stage_load_pretrained": True,
        "load_only_params": False,

        # Data paths
        "train_data": os.path.join(data_dir, "train_list.txt"),
        "val_data": os.path.join(data_dir, "val_list.txt"),
        "OOD_data": os.path.join(data_dir, "OOD_list.txt"),

        # PL-BERT settings - pointing to our Somali model
        "PLBERT_dir": "runs/plbert_so/packaged",
        "PLBERT_checkpoint": "runs/plbert_so/packaged/step_000001.pt",

        # Model parameters
        "decoder": {
            "type": "istftnet",
            "resblock_kernel_sizes": [3, 7, 11],
            "upsample_rates": [10, 6],
            "upsample_initial_channel": 512,
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_kernel_sizes": [20, 12]
        },

        # Training parameters
        "joint_epoch": 30,  # When to start SLM adversarial training
        "gradient_accumulation_steps": 1,
        "lr": {
            "scheduler": "CosineAnnealingLR",
            "initial": 1e-4,
            "warmup_steps": 200,
            "min_lr": 1e-6
        },

        # Loss weights
        "loss_weights": {
            "mel": 5.0,
            "dur": 1.0,
            "ce": 0.1,
            "norm": 1.0,
            "F0": 1.0,
            "energy": 0.1,
            "diff": 1.0,
            "slm": 1.0
        }
    }

    config_file = os.path.join(data_dir, "config_somali_ft.yml")

    # Write YAML config
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logging.info(f"Created config file: {config_file}")
    logging.info("Edit this file to adjust training parameters as needed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Somali dataset for StyleTTS2")
    parser.add_argument("--output_dir", type=str, default="data_styletts2",
                      help="Output directory for prepared data")
    parser.add_argument("--sample_rate", type=int, default=24000,
                      help="Target sample rate (StyleTTS2 uses 24kHz)")
    parser.add_argument("--max_duration", type=float, default=10.0,
                      help="Maximum audio duration in seconds")

    args = parser.parse_args()
    prepare_styletts2_dataset(args.output_dir, args.sample_rate, args.max_duration)