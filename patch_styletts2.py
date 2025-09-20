#!/usr/bin/env python
"""
Patch StyleTTS2 to handle PyTorch 2.6+ weights_only issue
"""
import os

def patch_models():
    """Patch the models.py file to fix torch.load issue."""

    models_file = "StyleTTS2/models.py"

    # Read the file
    with open(models_file, 'r') as f:
        content = f.read()

    # Replace torch.load calls to use weights_only=False
    if "weights_only=False" not in content:
        content = content.replace(
            "torch.load(model_path, map_location='cpu')",
            "torch.load(model_path, map_location='cpu', weights_only=False)"
        )
        content = content.replace(
            "torch.load(ASR_MODEL_PATH, map_location='cpu')",
            "torch.load(ASR_MODEL_PATH, map_location='cpu', weights_only=False)"
        )

        # Write back
        with open(models_file, 'w') as f:
            f.write(content)
        print(f"Patched {models_file}")
    else:
        print(f"{models_file} already patched")

def patch_train_finetune():
    """Patch the train_finetune.py file."""

    train_file = "StyleTTS2/train_finetune.py"

    # Read the file
    with open(train_file, 'r') as f:
        content = f.read()

    # Replace torch.load calls
    if "weights_only=False" not in content:
        content = content.replace(
            'torch.load(pretrained_model)',
            'torch.load(pretrained_model, weights_only=False)'
        )
        content = content.replace(
            'checkpoint = torch.load(pretrained_model, map_location=',
            'checkpoint = torch.load(pretrained_model, map_location='
        )

        # Find and replace all torch.load instances
        import re
        pattern = r'torch\.load\(([^)]+)\)'

        def add_weights_only(match):
            args = match.group(1)
            if 'weights_only' not in args:
                return f'torch.load({args}, weights_only=False)'
            return match.group(0)

        content = re.sub(pattern, add_weights_only, content)

        # Write back
        with open(train_file, 'w') as f:
            f.write(content)
        print(f"Patched {train_file}")
    else:
        print(f"{train_file} already patched")

def patch_utils():
    """Patch Utils files."""

    # Patch JDC
    jdc_file = "StyleTTS2/Utils/JDC/model.py"
    if os.path.exists(jdc_file):
        with open(jdc_file, 'r') as f:
            content = f.read()

        if "weights_only=False" not in content:
            content = content.replace(
                'torch.load(',
                'torch.load('
            )
            # Add weights_only=False to all torch.load calls
            import re
            pattern = r'torch\.load\(([^)]+)\)'

            def add_weights_only(match):
                args = match.group(1)
                if 'weights_only' not in args:
                    return f'torch.load({args}, weights_only=False)'
                return match.group(0)

            content = re.sub(pattern, add_weights_only, content)

            with open(jdc_file, 'w') as f:
                f.write(content)
            print(f"Patched {jdc_file}")

if __name__ == "__main__":
    print("Patching StyleTTS2 for PyTorch 2.6+ compatibility...")
    patch_models()
    patch_train_finetune()
    patch_utils()
    print("Patching complete!")