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
    """Update the config file with correct paths and format matching StyleTTS2."""

    config_path = "data_styletts2/config_somali_ft.yml"

    # Create proper StyleTTS2 config structure
    config = {
        'log_dir': 'Models/Somali',
        'save_freq': 5,
        'log_interval': 10,
        'device': 'cuda' if os.system("nvidia-smi > /dev/null 2>&1") == 0 else 'cpu',
        'epochs': 50,
        'batch_size': 4,  # Adjust based on GPU memory
        'max_len': 400,
        'pretrained_model': os.path.abspath('Models/LibriTTS/epochs_2nd_00020.pth'),
        'second_stage_load_pretrained': True,
        'load_only_params': True,

        # Paths to pre-trained components
        'F0_path': 'Utils/JDC/bst.t7',
        'ASR_config': 'Utils/ASR/config.yml',
        'ASR_path': 'Utils/ASR/epoch_00080.pth',
        'PLBERT_dir': os.path.abspath('runs/plbert_so/packaged'),  # Use our Somali PL-BERT!

        # Data parameters
        'data_params': {
            'train_data': os.path.abspath('data_styletts2/train_list.txt'),
            'val_data': os.path.abspath('data_styletts2/val_list.txt'),
            'root_path': os.path.abspath('data_styletts2/wavs'),
            'OOD_data': os.path.abspath('data_styletts2/OOD_list.txt'),
            'min_length': 50
        },

        # Preprocessing parameters
        'preprocess_params': {
            'sr': 24000,
            'spect_params': {
                'n_fft': 2048,
                'win_length': 1200,
                'hop_length': 300
            }
        },

        # Model parameters
        'model_params': {
            'multispeaker': False,  # Single speaker Somali
            'dim_in': 64,
            'hidden_dim': 512,
            'max_conv_dim': 512,
            'n_layer': 3,
            'n_mels': 80,
            'n_token': 178,  # Adjust if needed based on phoneme set
            'max_dur': 50,
            'style_dim': 128,
            'dropout': 0.2,

            # Decoder config
            'decoder': {
                'type': 'hifigan',
                'resblock_kernel_sizes': [3, 7, 11],
                'upsample_rates': [10, 5, 3, 2],
                'upsample_initial_channel': 512,
                'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                'upsample_kernel_sizes': [20, 10, 6, 4]
            },

            # SLM config
            'slm': {
                'model': 'microsoft/wavlm-base-plus',
                'sr': 16000,
                'hidden': 768,
                'nlayers': 13,
                'initial_channel': 64
            },

            # Diffusion config
            'diffusion': {
                'embedding_mask_proba': 0.1,
                'transformer': {
                    'num_layers': 3,
                    'num_heads': 8,
                    'head_features': 64,
                    'multiplier': 2
                },
                'dist': {
                    'sigma_data': 0.2,
                    'estimate_sigma_data': True,
                    'mean': -3.0,
                    'std': 1.0
                }
            }
        },

        # Loss parameters
        'loss_params': {
            'lambda_mel': 5.0,
            'lambda_gen': 1.0,
            'lambda_slm': 1.0,
            'lambda_mono': 1.0,
            'lambda_s2s': 1.0,
            'lambda_F0': 1.0,
            'lambda_norm': 1.0,
            'lambda_dur': 1.0,
            'lambda_ce': 20.0,
            'lambda_sty': 1.0,
            'lambda_diff': 1.0,
            'diff_epoch': 10,
            'joint_epoch': 30
        },

        # Optimizer parameters
        'optimizer_params': {
            'lr': 0.0001,
            'bert_lr': 0.00001,
            'ft_lr': 0.0001
        },

        # SLM adversarial parameters
        'slmadv_params': {
            'min_len': 400,
            'max_len': 500,
            'batch_percentage': 0.5,
            'iter': 10,
            'thresh': 5,
            'scale': 0.01,
            'sig': 1.5
        }
    }

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