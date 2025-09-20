#!/usr/bin/env python
"""
Find where the actual PL-BERT model weights are stored.
"""
import os
import torch
import glob

print("Searching for PL-BERT model files...")
print("=" * 70)

# Search patterns
search_dirs = [
    'runs/plbert_so/from_scratch',
    'runs/plbert_so/packaged',
    'runs/plbert_so',
    'runs',
    '.'
]

model_extensions = ['*.pt', '*.pth', '*.bin', '*.safetensors', '*.ckpt']

found_files = []

for dir_path in search_dirs:
    if os.path.exists(dir_path):
        print(f"\nChecking {dir_path}:")

        # List all files in this directory
        for ext in model_extensions:
            pattern = os.path.join(dir_path, ext)
            matches = glob.glob(pattern)
            if matches:
                for match in matches:
                    size_mb = os.path.getsize(match) / (1024 * 1024)
                    print(f"  Found: {match} ({size_mb:.2f} MB)")
                    found_files.append(match)

        # Also check subdirectories
        for subdir in os.listdir(dir_path):
            subdir_path = os.path.join(dir_path, subdir)
            if os.path.isdir(subdir_path):
                for ext in model_extensions:
                    pattern = os.path.join(subdir_path, ext)
                    matches = glob.glob(pattern)
                    if matches:
                        for match in matches:
                            size_mb = os.path.getsize(match) / (1024 * 1024)
                            print(f"  Found: {match} ({size_mb:.2f} MB)")
                            found_files.append(match)

print("\n" + "=" * 70)

if found_files:
    print(f"\nFound {len(found_files)} potential model files")

    # Try to load the largest file (likely the model)
    largest_file = max(found_files, key=os.path.getsize)
    print(f"\nLargest file: {largest_file}")
    print(f"Size: {os.path.getsize(largest_file) / (1024 * 1024):.2f} MB")

    print(f"\nTrying to load {largest_file}...")
    try:
        data = torch.load(largest_file, map_location='cpu', weights_only=False)

        if isinstance(data, dict):
            print(f"✓ Loaded successfully - it's a dictionary")
            print(f"  Keys: {list(data.keys())[:10]}")  # First 10 keys

            # Check if it contains model weights
            if 'model' in data or 'state_dict' in data or 'net' in data:
                print("  This looks like a checkpoint with model weights")

                # Extract the actual state dict
                if 'model' in data:
                    state_dict = data['model']
                elif 'state_dict' in data:
                    state_dict = data['state_dict']
                elif 'net' in data:
                    state_dict = data['net']
                else:
                    state_dict = data

                print(f"  State dict has {len(state_dict)} parameters")

                # Save to packaged directory
                output_path = 'runs/plbert_so/packaged/step_000001.pt'

                # Backup old file if exists
                if os.path.exists(output_path):
                    os.rename(output_path, output_path + '.bak')
                    print(f"\n  Backed up old file to {output_path}.bak")

                # Save just the state dict (not wrapped in another dict)
                torch.save(state_dict, output_path)
                print(f"\n✓ Saved model weights to {output_path}")

            elif all(isinstance(k, str) and '.' in k for k in list(data.keys())[:5]):
                # It's likely a state dict directly
                print("  This looks like a state dict directly")
                print(f"  Has {len(data)} parameters")
                print(f"  Sample keys: {list(data.keys())[:5]}")

                # Save to packaged directory
                output_path = 'runs/plbert_so/packaged/step_000001.pt'

                # Backup old file if exists
                if os.path.exists(output_path):
                    os.rename(output_path, output_path + '.bak')
                    print(f"\n  Backed up old file to {output_path}.bak")

                torch.save(data, output_path)
                print(f"\n✓ Saved model weights to {output_path}")

        else:
            print(f"Loaded data is of type: {type(data)}")
            print("This might not be a model file")

    except Exception as e:
        print(f"✗ Error loading file: {e}")

else:
    print("\n✗ No model files found")
    print("\nPlease check if the model was trained successfully")
    print("You may need to re-run the training or packaging step")