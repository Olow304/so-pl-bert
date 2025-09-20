#!/usr/bin/env python
"""
Fix the indentation error in meldataset.py
"""
import os

print("Fixing indentation error in meldataset.py...")
print("=" * 70)

meldataset_file = 'StyleTTS2/meldataset.py'

with open(meldataset_file, 'r') as f:
    lines = f.readlines()

print("Looking for the TextCleaner indentation issue...")

# Find the problematic area
for i, line in enumerate(lines[35:50], 35):
    print(f"{i:3}: {line.rstrip()}")

print("\n" + "=" * 70)
print("Fixing the indentation...")

# Fix the indentation issue
for i, line in enumerate(lines):
    if i >= 40 and i <= 45:
        # Check if this is around the except block
        if 'except KeyError:' in line:
            # Make sure the next line is properly indented
            if i+1 < len(lines):
                next_line = lines[i+1]
                if '# print(text)' in next_line or 'print(text)' in next_line:
                    # Ensure it's indented correctly (should have more indent than except)
                    indent_level = len(line) - len(line.lstrip())
                    lines[i+1] = ' ' * (indent_level + 4) + next_line.lstrip()

                    # Also make sure the return statement after is correctly indented
                    if i+2 < len(lines) and 'return indexes' in lines[i+2]:
                        lines[i+2] = ' ' * indent_level + lines[i+2].lstrip()
                elif 'return indexes' in next_line:
                    # The return is directly after except, add proper indentation
                    indent_level = len(line) - len(line.lstrip())
                    lines[i+1] = ' ' * (indent_level + 4) + next_line.lstrip()
                else:
                    # Add a pass statement if there's nothing after except
                    indent_level = len(line) - len(line.lstrip())
                    lines.insert(i+1, ' ' * (indent_level + 4) + 'pass  # Handle KeyError\n')

# Write the fixed file
with open(meldataset_file, 'w') as f:
    f.writelines(lines)

print("✓ Fixed indentation")

# Verify the fix
print("\n" + "=" * 70)
print("Verifying the fix...")

os.chdir('StyleTTS2')
import sys
sys.path.insert(0, '.')

try:
    from meldataset import build_dataloader
    print("✓ meldataset imports successfully")
except IndentationError as e:
    print(f"✗ Still has indentation error: {e}")
    print("\nManually fixing...")

    # Read again and do a more aggressive fix
    os.chdir('..')
    with open(meldataset_file, 'r') as f:
        content = f.read()

    # Replace the problematic section
    old_section = """            except KeyError:
                # print(text)
        return indexes"""

    new_section = """            except KeyError:
                pass  # print(text) - commented out
        return indexes"""

    if old_section in content:
        content = content.replace(old_section, new_section)
    else:
        # Try another pattern
        import re
        content = re.sub(
            r'except KeyError:\s*\n\s*#\s*print\(text\)\s*\n\s*return indexes',
            'except KeyError:\n                pass  # print(text) commented\n        return indexes',
            content
        )

    with open(meldataset_file, 'w') as f:
        f.write(content)

    print("✓ Applied aggressive fix")

    # Test again
    os.chdir('StyleTTS2')
    from meldataset import build_dataloader
    print("✓ Now imports successfully")

print("\n" + "=" * 70)
print("Now run: python minimal_train.py")