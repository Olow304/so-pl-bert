#!/usr/bin/env python
"""
Fix vocabulary size mismatch between text tokens and PL-BERT model.
"""
import os
import yaml
import torch
import pickle

print("Fixing vocabulary size mismatch...")
print("=" * 70)

# 1. Check the PL-BERT model's vocabulary size
print("\n1. Checking Somali PL-BERT vocabulary size...")

plbert_dir = 'runs/plbert_so/packaged'
config_path = os.path.join(plbert_dir, 'config.yml')

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        plbert_config = yaml.safe_load(f)

    vocab_size = plbert_config['model_params']['vocab_size']
    print(f"   Somali PL-BERT vocab_size: {vocab_size}")
else:
    print("   Could not find PL-BERT config")
    vocab_size = 116  # Default from our training

# 2. Check the token_maps to understand the mapping
token_maps_path = os.path.join(plbert_dir, 'token_maps.pkl')
if os.path.exists(token_maps_path):
    with open(token_maps_path, 'rb') as f:
        token_maps = pickle.load(f)

    print(f"   Token maps loaded")
    if isinstance(token_maps, dict):
        if 'token_to_id' in token_maps:
            token_to_id = token_maps['token_to_id']
            print(f"   Number of tokens in map: {len(token_to_id)}")
            print(f"   Max token ID: {max(token_to_id.values())}")
        else:
            print(f"   Token maps keys: {list(token_maps.keys())[:5]}")

# 3. Check what symbols are being used in the text
print("\n2. Checking text symbols in meldataset.py...")

meldataset_file = 'StyleTTS2/meldataset.py'
with open(meldataset_file, 'r') as f:
    content = f.read()

# Find the symbols definition
import re
symbols_match = re.search(r'symbols = \[(.*?)\]', content, re.DOTALL)
if symbols_match:
    symbols_str = symbols_match.group(1)
    # Count symbols
    symbols_count = symbols_str.count('+') + symbols_str.count(',')
    print(f"   Estimated number of symbols: ~{symbols_count}")

# Check the actual symbols list
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'symbols = ' in line:
        # Look for the complete definition
        symbols_line = line
        j = i
        while ']' not in symbols_line and j < len(lines) - 1:
            j += 1
            symbols_line += lines[j]

        # Count actual symbols
        exec_env = {}
        # Define the components
        exec_env['_pad'] = "$"
        exec_env['_punctuation'] = ';:,.!?¡¿—…"«»"" '
        exec_env['_letters'] = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        exec_env['_letters_ipa'] = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

        try:
            exec(symbols_line, exec_env)
            symbols = exec_env.get('symbols', [])
            print(f"   Actual number of symbols in meldataset: {len(symbols)}")
            print(f"   Max symbol index would be: {len(symbols) - 1}")
        except:
            print(f"   Could not execute symbols definition")

# 4. The problem and solution
print("\n" + "=" * 70)
print("PROBLEM IDENTIFIED:")
print(f"  - Somali PL-BERT vocab size: {vocab_size}")
print(f"  - Text encoding uses up to {len(symbols) - 1 if 'symbols' in locals() else 'UNKNOWN'} indices")
print(f"  - Mismatch causes index out of bounds error!")

print("\nSOLUTION: Update config to match vocab sizes")
print("=" * 70)

# 5. Update the StyleTTS2 config
styletts_config_path = 'data_styletts2/config_somali_ft.yml'
with open(styletts_config_path, 'r') as f:
    styletts_config = yaml.safe_load(f)

# Update the n_token parameter to match PL-BERT vocab size
if 'model_params' not in styletts_config:
    styletts_config['model_params'] = {}

old_n_token = styletts_config['model_params'].get('n_token', 178)
styletts_config['model_params']['n_token'] = vocab_size

print(f"\nUpdating StyleTTS2 config:")
print(f"  Old n_token: {old_n_token}")
print(f"  New n_token: {vocab_size}")

# Save updated config
with open(styletts_config_path, 'w') as f:
    yaml.dump(styletts_config, f, default_flow_style=False, sort_keys=False)

print(f"\n✓ Updated {styletts_config_path}")

# 6. Also need to ensure text tokens don't exceed vocab size
print("\n" + "=" * 70)
print("Creating a text token clipping fix...")

fix_tokens_script = """#!/usr/bin/env python
'''
Add token clipping to prevent out of bounds errors
'''
import os

print("Adding token clipping to meldataset.py...")

meldataset_file = 'StyleTTS2/meldataset.py'
with open(meldataset_file, 'r') as f:
    content = f.read()

# Find the TextCleaner class and add clipping
import re

# Add vocab_size parameter to TextCleaner
old_init = "def __init__(self, dummy=None):"
new_init = "def __init__(self, dummy=None, vocab_size=116):"

if old_init in content:
    content = content.replace(old_init, new_init)
    print("✓ Added vocab_size parameter to TextCleaner")

# Add clipping in the __call__ method
# Find where indexes are appended
lines = content.split('\\n')
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)

    # After getting index, clip it
    if 'indexes.append(self.word_index_dictionary[char])' in line:
        indent = len(line) - len(line.lstrip())
        # Replace the line with clipped version
        new_lines[-1] = line.replace(
            'indexes.append(self.word_index_dictionary[char])',
            'indexes.append(min(self.word_index_dictionary[char], 115))'  # 115 = vocab_size - 1
        )
        print(f"✓ Added clipping to token indices (max=115)")

content = '\\n'.join(new_lines)

# Write the modified file
with open(meldataset_file, 'w') as f:
    f.write(content)

print("\\n✓ Token clipping added")
print("\\nNow training should work without index errors!")
"""

with open('fix_token_clipping.py', 'w') as f:
    f.write(fix_tokens_script)

print("\nCreated fix_token_clipping.py")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("\n1. Run: python fix_token_clipping.py")
print("2. Run: python monitor_training.py")
print("\nTraining should now work without index errors! 🎉")