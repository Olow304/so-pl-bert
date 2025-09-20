#!/usr/bin/env python
"""
Test if the data can be loaded properly.
"""
import os
import sys
sys.path.append('StyleTTS2')

def test_data_loading():
    """Test loading the training data."""

    train_file = "data_styletts2/train_list.txt"

    print("Testing data loading...")

    # Read train list
    with open(train_file, 'r') as f:
        lines = f.readlines()

    print(f"Found {len(lines)} training samples")

    # Check first few samples
    errors = []
    for i, line in enumerate(lines[:5]):
        parts = line.strip().split('|')
        if len(parts) != 3:
            errors.append(f"Line {i}: Wrong format - {line}")
            continue

        audio_file, phonemes, speaker = parts
        audio_path = os.path.join("data_styletts2/wavs", audio_file)

        if not os.path.exists(audio_path):
            errors.append(f"Line {i}: Audio not found - {audio_path}")
        else:
            size = os.path.getsize(audio_path)
            print(f"✓ Sample {i}: {audio_file} ({size} bytes), speaker={speaker}")
            print(f"  Phonemes: {phonemes[:50]}...")

    if errors:
        print("\nErrors found:")
        for e in errors:
            print(f"  {e}")
    else:
        print("\n✓ Data format looks good!")

    # Try importing the training modules
    try:
        from meldataset import build_dataloader
        print("\n✓ Can import meldataset")
    except ImportError as e:
        print(f"\n✗ Cannot import meldataset: {e}")

    # Check config
    import yaml
    config_path = "data_styletts2/config_somali_ft.yml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"\n✓ Config loaded successfully")
        print(f"  Batch size: {config.get('batch_size', 'Not set')}")
        print(f"  Epochs: {config.get('epochs', 'Not set')}")
        print(f"  Device: {config.get('device', 'Not set')}")

if __name__ == "__main__":
    test_data_loading()