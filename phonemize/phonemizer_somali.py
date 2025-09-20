#!/usr/bin/env python
"""
Fallback Somali grapheme‑to‑phoneme (G2P) module.  When eSpeak NG does not
support Somali or is unavailable, this simple rule‑based phonemizer
approximates Somali phonology by mapping letters and digraphs to phones
according to the linguistic description of the language【275631273323948†L200-L208】【275631273323948†L288-L292】.  It covers long vowels, geminate
consonants, uvular and pharyngeal consonants, and common digraphs (dh, kh,
sh).  The output is a list of phoneme tokens corresponding to the input
string.

Note that this G2P is highly simplified and should be replaced with a
learned or more sophisticated model for production use.  It serves as a
fallback to ensure that the PL‑BERT model can be trained even if eSpeak NG
cannot produce Somali phonemes.
"""
from __future__ import annotations

import re
from typing import List

PHONEME_MAP = {
    # vowels (short)
    "a": "a", "e": "e", "i": "i", "o": "o", "u": "u",
    # consonants
    "b": "b", "t": "t", "j": "dʒ", "x": "ħ", # x = pharyngeal fricative【275631273323948†L200-L208】
    "kh": "x", # uvular fricative
    "d": "d", "dh": "ð", "r": "r", "s": "s", "sh": "ʃ", "c": "ʕ",  # c = voiced pharyngeal【275631273323948†L200-L208】
    "g": "ɡ", "f": "f", "q": "q", "k": "k", "l": "l", "m": "m", "n": "n", "p": "p", "w": "w", "y": "j",
}

# Map long vowels (aa, ee, ii, oo, uu) to lengthened phones
LONG_VOWELS = {
    "aa": "aː", "ee": "eː", "ii": "iː", "oo": "oː", "uu": "uː",
}

# Digraphs for consonants
DIGRAPHS = {"dh", "kh", "sh"}


def grapheme_to_phonemes(word: str) -> List[str]:
    """Convert a Somali word into a list of phoneme tokens."""
    word = word.lower()
    phonemes = []
    i = 0
    while i < len(word):
        # handle long vowels
        if i + 1 < len(word) and word[i:i+2] in LONG_VOWELS:
            phonemes.append(LONG_VOWELS[word[i:i+2]])
            i += 2
            continue
        # handle consonant digraphs
        if i + 1 < len(word) and word[i:i+2] in DIGRAPHS:
            phonemes.append(PHONEME_MAP.get(word[i:i+2], word[i:i+2]))
            i += 2
            continue
        char = word[i]
        # gemination: if same letter repeated, duplicate the phoneme
        if i + 1 < len(word) and word[i+1] == char:
            # treat as geminate; double the phoneme
            phone = PHONEME_MAP.get(char, char)
            phonemes.extend([phone, phone])
            i += 2
            continue
        # simple mapping
        phonemes.append(PHONEME_MAP.get(char, char))
        i += 1
    return phonemes


def phonemize_sentence(sentence: str) -> tuple[str, str]:
    """Phonemize a sentence, returning (phonemes_string, graphemes_string).  Words
    are separated by spaces in the input.  Graphemes are tokenised as
    characters separated by spaces with an underscore marking the end of each
    word.  Digraphs and long vowels are treated as single tokens to align
    with the phoneme tokens.
    """
    words = sentence.strip().split()
    phone_tokens: list[str] = []
    grapheme_tokens: list[str] = []
    for word in words:
        # generate phonemes
        phs = grapheme_to_phonemes(word)
        phone_tokens.extend(phs + ["_"])  # use '_' as word separator
        # grapheme tokens: treat digraphs and long vowels as single tokens
        i = 0
        while i < len(word):
            if i + 1 < len(word) and word[i:i+2] in LONG_VOWELS:
                grapheme_tokens.append(word[i:i+2])
                i += 2
                continue
            if i + 1 < len(word) and word[i:i+2] in DIGRAPHS:
                grapheme_tokens.append(word[i:i+2])
                i += 2
                continue
            # gemination: handle double letters as two separate tokens (for grapheme level)
            grapheme_tokens.append(word[i])
            i += 1
        grapheme_tokens.append("_")
    # Remove trailing separator
    if phone_tokens and phone_tokens[-1] == "_":
        phone_tokens.pop()
    if grapheme_tokens and grapheme_tokens[-1] == "_":
        grapheme_tokens.pop()
    return " ".join(phone_tokens), " ".join(grapheme_tokens)


if __name__ == "__main__":
    # Simple CLI test
    import sys
    for sentence in sys.stdin:
        p, g = phonemize_sentence(sentence.strip())
        print({"phonemes": p, "graphemes": g})