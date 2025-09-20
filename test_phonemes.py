#!/usr/bin/env python
"""
Test script to validate phonemization pipeline and PL-BERT encoding.
This shows exactly what StyleTTS2 would receive.
"""
import pickle
import sys
import os
sys.path.append('phonemize')

from phonemizer_somali import phonemize_sentence

def test_phonemization():
    """Test the phonemization pipeline with sample Somali text."""

    # Test sentences
    test_sentences = [
        "Soo dhawoow",
        "Sidee tahay?",
        "Waan fiicanahay, mahadsanid",
        "Magacaygu waa Cali",
        "Subax wanaagsan",
        "Galab wanaagsan",
        "Habeen wanaagsan",
        "Iska warran",
        "Waa maxay magacaaga?",
        "Ka bax halkan"
    ]

    print("=" * 70)
    print("PHONEMIZATION TEST - What StyleTTS2 will see")
    print("=" * 70)

    # Load token maps to show encoding
    if os.path.exists("phonemize/token_maps.pkl"):
        with open("phonemize/token_maps.pkl", "rb") as f:
            token_to_id = pickle.load(f)
        id_to_token = {v: k for k, v in token_to_id.items()}
        print(f"\nVocabulary size: {len(token_to_id)} tokens")
        print(f"Special tokens: <pad>={token_to_id.get('<pad>')}, <unk>={token_to_id.get('<unk>')}, <mask>={token_to_id.get('<mask>')}")
    else:
        token_to_id = None
        id_to_token = None
        print("\nWarning: token_maps.pkl not found - run 'make data' first")

    print("\n" + "=" * 70)

    for sentence in test_sentences:
        print(f"\nOriginal: {sentence}")

        # Phonemize
        phonemes, graphemes = phonemize_sentence(sentence)
        print(f"Phonemes: {phonemes}")
        print(f"Graphemes: {graphemes}")

        # Show token encoding if available
        if token_to_id:
            phoneme_tokens = phonemes.split()
            token_ids = []
            unknown_tokens = []

            for token in phoneme_tokens:
                if token in token_to_id:
                    token_ids.append(token_to_id[token])
                else:
                    token_ids.append(token_to_id.get("<unk>", 0))
                    unknown_tokens.append(token)

            print(f"Token IDs: {token_ids}")
            if unknown_tokens:
                print(f"⚠️  Unknown tokens (mapped to <unk>): {unknown_tokens}")

            # Verify decoding
            decoded = [id_to_token.get(tid, "?") for tid in token_ids]
            print(f"Decoded: {' '.join(decoded)}")

        print("-" * 50)

    # Test for common issues
    print("\n" + "=" * 70)
    print("VALIDATION CHECKS:")
    print("=" * 70)

    # Check 1: Consonant gemination
    test = "maallin"  # Should have doubled 'l'
    ph, _ = phonemize_sentence(test)
    if "lː" in ph or "l l" in ph or any(c*2 in ph for c in 'lmn'):
        print(f"✓ Gemination working: '{test}' → '{ph}'")
    else:
        print(f"⚠️  Gemination issue: '{test}' → '{ph}' (expected doubled consonant)")

    # Check 2: Long vowels
    test = "soomaaliya"  # Should have long vowels
    ph, _ = phonemize_sentence(test)
    if "ː" in ph or "aa" in ph or "oo" in ph:
        print(f"✓ Long vowels working: '{test}' → '{ph}'")
    else:
        print(f"⚠️  Long vowel issue: '{test}' → '{ph}' (expected vowel length markers)")

    # Check 3: Pharyngeal/uvular sounds
    test = "cali xasan qaran"  # c=pharyngeal, x=pharyngeal, q=uvular
    ph, _ = phonemize_sentence(test)
    print(f"✓ Special consonants: '{test}' → '{ph}'")

    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("- This is what StyleTTS2 will receive as input")
    print("- Each phoneme becomes a token ID that PL-BERT processes")
    print("- StyleTTS2 uses these embeddings to generate speech")
    print("=" * 70)

if __name__ == "__main__":
    test_phonemization()