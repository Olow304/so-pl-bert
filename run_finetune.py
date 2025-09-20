#!/usr/bin/env python
"""
Run StyleTTS2 fine-tuning with Somali PL-BERT.
This script properly configures and launches the training.
"""
import os
import sys
import subprocess
import yaml

def update_config():
    """Update the config file with correct paths."""

    config_path = "data_styletts2/config_somali_ft.yml"

    # Read existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update paths to be absolute
    config['train_data'] = os.path.abspath("data_styletts2/train_list.txt")
    config['val_data'] = os.path.abspath("data_styletts2/val_list.txt")
    config['OOD_data'] = os.path.abspath("data_styletts2/OOD_list.txt")
    config['data_path'] = os.path.abspath("data_styletts2/wavs")

    # Update model paths
    config['PLBERT_dir'] = os.path.abspath("runs/plbert_so/packaged")
    config['pretrained_model'] = os.path.abspath("Models/LibriTTS/epochs_2nd_00020.pth")
    config['log_dir'] = os.path.abspath("Models/Somali")

    # Ensure proper training parameters
    config['batch_size'] = 4  # Adjust based on GPU memory
    config['epochs'] = 50
    config['save_freq'] = 1
    config['log_interval'] = 10
    config['joint_epoch'] = 30  # Start adversarial training

    # Add missing keys that StyleTTS2 expects
    config['first_stage_path'] = None
    config['second_stage_path'] = config['pretrained_model']
    config['load_only_params'] = False
    config['F0_path'] = "Utils/JDC"
    config['ASR_path'] = "Utils/ASR"
    config['ASR_config'] = "Utils/ASR/config.yml"

    # Data configuration
    config['max_len'] = 400  # ~5 seconds at 24kHz
    config['min_length'] = 50
    config['batch_percentage'] = 0.5
    config['device'] = 'cuda' if os.system("nvidia-smi > /dev/null 2>&1") == 0 else 'cpu'

    # Training strategy
    config['TMA_epoch'] = 5
    config['TMA_CEloss'] = False
    config['diff_epoch'] = 10
    config['joint_epoch'] = 30

    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Updated config saved to {config_path}")
    return config_path

def run_training():
    """Run the actual StyleTTS2 training."""

    # Update configuration
    config_path = update_config()

    # Change to StyleTTS2 directory
    if not os.path.exists("StyleTTS2"):
        print("ERROR: StyleTTS2 directory not found!")
        print("Please run: bash setup_styletts2.sh first")
        return

    os.chdir("StyleTTS2")

    # Run training
    cmd = [
        "python", "train_finetune.py",
        "--config_path", f"../{config_path}"
    ]

    print("=" * 70)
    print("Starting StyleTTS2 fine-tuning with Somali PL-BERT")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"PL-BERT: runs/plbert_so/packaged")
    print(f"Training data: data_styletts2/train_list.txt")
    print("=" * 70)

    # Execute training
    subprocess.run(cmd)

if __name__ == "__main__":
    run_training()