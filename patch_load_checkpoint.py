#!/usr/bin/env python
"""
Patch the load_checkpoint function to handle BERT size mismatch.
"""
import os
import re

def patch_models_load_checkpoint():
    """Patch models.py to skip incompatible BERT weights."""

    models_file = "StyleTTS2/models.py"

    # Read the file
    with open(models_file, 'r') as f:
        content = f.read()

    # Find the load_checkpoint function
    func_start = content.find("def load_checkpoint(")
    if func_start == -1:
        print("Could not find load_checkpoint function")
        return

    # Find the end of the function (next def or class)
    func_end = content.find("\ndef ", func_start + 1)
    if func_end == -1:
        func_end = content.find("\nclass ", func_start + 1)
    if func_end == -1:
        func_end = len(content)

    # Replace the function with a patched version
    new_function = '''def load_checkpoint(model, optimizer, path, load_only_params=False, ignore_layers=[], is_distributed=False):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    if 'model' in checkpoint:
        raw_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint['state_dict']
    else:
        raw_state_dict = checkpoint

    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.startswith('module.'):
            state_dict[k[7:]] = v
        else:
            state_dict[k] = v

    model_dict = model.state_dict()
    model_dict = {k: v for k, v in model_dict.items() if k not in ignore_layers}

    # Handle BERT separately to avoid size mismatch
    for key in list(state_dict.keys()):
        if 'bert' in key.lower():
            # Check if shapes match
            if key in model_dict:
                if model_dict[key].shape != state_dict[key].shape:
                    print(f"Skipping {key} due to shape mismatch: checkpoint {state_dict[key].shape} vs model {model_dict[key].shape}")
                    del state_dict[key]

    if is_distributed:
        for key in model:
            if key == 'bert_encoder':
                continue  # Skip BERT encoder
            try:
                model[key].load_state_dict({k: state_dict[k] for k in state_dict if key in k}, strict=False)
            except Exception as e:
                print(f"Warning: Could not load {key}: {e}")
    else:
        for key in model:
            if key == 'bert_encoder':
                continue  # Skip BERT encoder
            try:
                model[key].load_state_dict(state_dict, strict=False)
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    # Try loading only compatible layers
                    compatible_state = {}
                    for k, v in state_dict.items():
                        if k in model[key].state_dict():
                            if model[key].state_dict()[k].shape == v.shape:
                                compatible_state[k] = v
                            else:
                                print(f"Skipping {k} in {key} due to shape mismatch")
                    if compatible_state:
                        model[key].load_state_dict(compatible_state, strict=False)
                        print(f"Loaded {len(compatible_state)} compatible parameters for {key}")
                else:
                    print(f"Error loading {key}: {e}")

    if not load_only_params and 'optimizer' in checkpoint and optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print("Could not load optimizer state dict, using fresh optimizer")

    if not load_only_params:
        start_epoch = checkpoint.get('epoch', 1)
        iters = checkpoint.get('iters', 0) + 1
    else:
        start_epoch = 1
        iters = 0

    return model, optimizer, start_epoch, iters
'''

    # Replace the function
    content = content[:func_start] + new_function + content[func_end:]

    # Write back
    with open(models_file, 'w') as f:
        f.write(content)

    print(f"Patched {models_file} to handle BERT size mismatch")

if __name__ == "__main__":
    print("Patching load_checkpoint to skip incompatible BERT weights...")
    patch_models_load_checkpoint()
    print("Patching complete!")