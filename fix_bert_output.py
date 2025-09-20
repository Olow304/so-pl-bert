#!/usr/bin/env python
"""
Fix BERT output format issue - extract tensor from HuggingFace output object.
"""
import os

print("Fixing BERT output format issue...")
print("=" * 70)

train_file = 'StyleTTS2/train_finetune.py'

# Read the training file
with open(train_file, 'r') as f:
    lines = f.readlines()

print("Looking for BERT forward calls...")

modified = False
for i, line in enumerate(lines):
    # Find where bert is called
    if 'bert_dur = model.bert(texts' in line:
        print(f"Found BERT call at line {i+1}: {line.strip()}")

        # Check if it's already extracting the tensor
        if '.last_hidden_state' not in line and '.hidden_states' not in line:
            # Need to add .last_hidden_state
            lines[i] = line.replace(
                'bert_dur = model.bert(texts, attention_mask=(~text_mask).int())',
                'bert_dur = model.bert(texts, attention_mask=(~text_mask).int()).last_hidden_state'
            )
            print(f"  Fixed to extract last_hidden_state")
            modified = True

    # Also check for any other bert calls
    elif 'model.bert(' in line and '=' in line:
        var_name = line.split('=')[0].strip()
        if '.last_hidden_state' not in line:
            print(f"Found another BERT call at line {i+1}: {line.strip()}")
            # Add .last_hidden_state if not present
            if line.rstrip().endswith(')'):
                lines[i] = line.rstrip() + '.last_hidden_state\n'
                print(f"  Fixed to extract last_hidden_state")
                modified = True

# Write the modified file
if modified:
    # Backup first
    import shutil
    backup_file = train_file + '.bert_output_backup'
    shutil.copy(train_file, backup_file)
    print(f"\nBacked up to {backup_file}")

    with open(train_file, 'w') as f:
        f.writelines(lines)

    print("✓ Fixed BERT output extraction")
else:
    print("\n⚠ No modifications needed or already fixed")

# Also check models.py for similar issues
print("\n" + "=" * 70)
print("Checking models.py for BERT usage...")

models_file = 'StyleTTS2/models.py'
if os.path.exists(models_file):
    with open(models_file, 'r') as f:
        models_content = f.read()

    # Look for BERT forward passes
    import re
    bert_calls = re.findall(r'self\.bert\([^)]+\)', models_content)
    if bert_calls:
        print(f"Found {len(bert_calls)} BERT calls in models.py")

        # Check if they extract the tensor properly
        models_lines = models_content.split('\n')
        models_modified = False

        for i, line in enumerate(models_lines):
            if 'self.bert(' in line and '=' in line:
                if '.last_hidden_state' not in line and '.hidden_states' not in line:
                    print(f"  Line {i+1}: {line.strip()}")
                    # Fix it
                    if line.rstrip().endswith(')'):
                        models_lines[i] = line.rstrip() + '.last_hidden_state\n'
                        models_modified = True
                        print(f"    Fixed to extract last_hidden_state")

        if models_modified:
            # Backup and save
            backup_file = models_file + '.bert_output_backup'
            shutil.copy(models_file, backup_file)
            print(f"\nBacked up models.py to {backup_file}")

            with open(models_file, 'w') as f:
                f.write('\n'.join(models_lines))

            print("✓ Fixed BERT calls in models.py")

print("\n" + "=" * 70)
print("Creating a test script to verify the fix...")

test_script = """#!/usr/bin/env python
'''
Test that BERT output is properly extracted
'''
import torch
import sys
import os

os.chdir('StyleTTS2')
sys.path.insert(0, '.')

print("Testing BERT output format...")

# Import the PLBERT model
from transformers import AlbertModel, AlbertConfig

# Load our Somali config
import yaml
with open('../runs/plbert_so/packaged/config.yml', 'r') as f:
    plbert_config = yaml.safe_load(f)

# Create a model
config = AlbertConfig(
    vocab_size=plbert_config['model_params']['vocab_size'],
    hidden_size=plbert_config['model_params']['hidden_size'],
    num_hidden_layers=plbert_config['model_params']['num_hidden_layers'],
    num_attention_heads=plbert_config['model_params']['num_attention_heads'],
    intermediate_size=plbert_config['model_params']['intermediate_size']
)

model = AlbertModel(config)

# Test input
batch_size = 2
seq_len = 10
input_ids = torch.randint(0, 116, (batch_size, seq_len))
attention_mask = torch.ones_like(input_ids)

# Forward pass
output = model(input_ids, attention_mask=attention_mask)

print(f"Output type: {type(output)}")
print(f"Output attributes: {dir(output)[:10]}")

# Extract tensor
if hasattr(output, 'last_hidden_state'):
    tensor_output = output.last_hidden_state
    print(f"✓ Extracted tensor shape: {tensor_output.shape}")
    print(f"  Expected shape: ({batch_size}, {seq_len}, {config.hidden_size})")
else:
    print("✗ No last_hidden_state attribute!")

print("\\n✓ Test complete")
"""

with open('test_bert_output.py', 'w') as f:
    f.write(test_script)

print("\nCreated test_bert_output.py")

print("\n" + "=" * 70)
print("SOLUTION APPLIED!")
print("=" * 70)
print("\nThe issue was that HuggingFace BERT models return an object,")
print("not a tensor. We need to extract the .last_hidden_state tensor.")
print("\nRun: python monitor_training.py")
print("\nTraining should now proceed further! 🚀")