#!/usr/bin/env python
"""
Patch train_finetune.py to skip loading BERT weights from pre-trained model.
"""
import os

def patch_train_finetune():
    """Patch to skip BERT weight loading since we have different vocab size."""

    train_file = "StyleTTS2/train_finetune.py"

    # Read the file
    with open(train_file, 'r') as f:
        lines = f.readlines()

    # Find and modify the checkpoint loading section
    modified = False
    for i, line in enumerate(lines):
        # Look for the load_checkpoint call
        if "model, optimizer, start_epoch, iters = load_checkpoint" in line:
            # Add a comment and modify to skip BERT
            lines[i] = "    # Skip loading BERT weights - we use our own Somali BERT\n"
            lines[i] += "    skip_bert = True  # Add this flag\n"
            lines[i] += "    model, optimizer, start_epoch, iters = load_checkpoint(model, optimizer, config['pretrained_model'],\n"
            modified = True
            break

    if modified:
        with open(train_file, 'w') as f:
            f.writelines(lines)
        print(f"Patched {train_file} to skip BERT loading")

    # Now patch the models.py file to handle skip_bert
    models_file = "StyleTTS2/models.py"

    with open(models_file, 'r') as f:
        content = f.read()

    # Find the load_checkpoint function and modify it
    if "def load_checkpoint(" in content:
        # Add logic to skip BERT loading
        new_load_checkpoint = '''
def load_checkpoint(model, optimizer, path, load_only_params=False, ignore_layers=[], is_distributed=False):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    raw_state_dict = checkpoint['model' if 'model' in checkpoint else 'model_state_dict']
    state_dict = {}

    # Skip BERT layers for different vocab size
    for k, v in raw_state_dict.items():
        # Skip BERT/PLBERT related weights
        if any(bert_key in k for bert_key in ['bert', 'BERT', 'plbert', 'PLBERT']):
            print(f"Skipping BERT layer: {k}")
            continue

        if k.startswith('module.'):
            k = k.replace('module.', '')
        state_dict[k] = v

    model_dict = model.state_dict()

    # Filter out mismatched keys
    pretrained_dict = {}
    for k, v in state_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                pretrained_dict[k] = v
            else:
                print(f"Shape mismatch for {k}: model {model_dict[k].shape} vs checkpoint {v.shape}")
        else:
            print(f"Key {k} not found in model")

    # Load the filtered dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    if not load_only_params and 'optimizer' in checkpoint and optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print("Could not load optimizer state")

    if not load_only_params:
        start_epoch = checkpoint.get('epoch', 1)
        iters = checkpoint.get('iters', 0)
    else:
        start_epoch = 1
        iters = 0

    return model, optimizer, start_epoch, iters
'''

        # Replace the old function
        import re
        pattern = r'def load_checkpoint\([^{]+\{[^}]+\}[^}]+\}'

        # Simple replacement - find function start
        start_idx = content.find("def load_checkpoint(")
        if start_idx != -1:
            # Find the next function or end of file
            next_func = content.find("\ndef ", start_idx + 1)
            if next_func == -1:
                next_func = len(content)

            # Replace the function
            content = content[:start_idx] + new_load_checkpoint + content[next_func:]

            with open(models_file, 'w') as f:
                f.write(content)
            print(f"Patched {models_file} to skip BERT weights")

if __name__ == "__main__":
    print("Patching to skip BERT weight loading...")
    patch_train_finetune()
    print("Patching complete!")