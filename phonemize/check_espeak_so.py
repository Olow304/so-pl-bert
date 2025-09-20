#!/usr/bin/env python
"""
Check whether eSpeak NG supports Somali.  The script runs `espeak-ng
--voices` and searches for a voice code matching Somali (`so`).  It returns
exit code 0 if a Somali voice is available, 1 if not supported, and 2 if
eSpeak NG is not installed.  Use this script to detect at runtime whether
eSpeak can be used for phonemization【371023369836432†L88-L104】.

Usage:
    python check_espeak_so.py && echo "Somali voice available" || echo "Somali voice unavailable"

"""
import subprocess
import sys

def main():
    try:
        out = subprocess.check_output(["espeak-ng", "--voices"], stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        print("espeak-ng not installed", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError:
        print("Failed to run espeak-ng", file=sys.stderr)
        return 2
    for line in out.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        # Format: index language code gender ... voice name
        lang = parts[1] if len(parts) > 1 else ""
        if lang.lower() == "so":
            return 0
    return 1

if __name__ == "__main__":
    sys.exit(main())