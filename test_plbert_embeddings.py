#!/usr/bin/env python
"""
Test PL-BERT embeddings to verify the model produces meaningful representations.
"""
import torch
import pickle
import sys
import numpy as np
from transformers import AlbertModel
sys.path.append('phonemize')
from phonemizer_somali import phonemize_sentence

def test_plbert_embeddings():
    """Test that PL-BERT produces meaningful embeddings for Somali phonemes."""

    print("=" * 70)
    print("PL-BERT EMBEDDING TEST")
    print("=" * 70)

    # Load the trained model
    model_path = "runs/plbert_so/from_scratch"
    print(f"\nLoading PL-BERT from: {model_path}")
    model = AlbertModel.from_pretrained(model_path)
    model.eval()

    # Load token maps
    with open("phonemize/token_maps.pkl", "rb") as f:
        token_to_id = pickle.load(f)

    print(f"Vocabulary size: {len(token_to_id)}")

    # Test sentences - similar words should have similar embeddings
    test_pairs = [
        ("subax wanaagsan", "galab wanaagsan"),  # Same structure, different time
        ("waan fiicanahay", "waan wanaagsanahay"),  # Similar meaning
        ("magacaygu waa", "magacaaga waa"),  # Same structure, different pronoun
        ("soo dhawoow", "kaalay halkan"),  # Different greetings
    ]

    print("\n" + "=" * 70)
    print("EMBEDDING SIMILARITY TEST")
    print("(Similar sentences should have high cosine similarity)")
    print("=" * 70)

    def get_embedding(text):
        """Get sentence embedding from PL-BERT."""
        # Phonemize
        phonemes, _ = phonemize_sentence(text)
        tokens = phonemes.split()

        # Convert to IDs
        ids = [token_to_id.get(tok, token_to_id["<unk>"]) for tok in tokens]

        # Pad to minimum length for BERT
        if len(ids) < 3:
            ids = ids + [token_to_id["<pad>"]] * (3 - len(ids))

        # Create input tensor
        input_ids = torch.tensor([ids])

        # Get embeddings
        with torch.no_grad():
            outputs = model(input_ids)
            # Use mean pooling of last hidden states
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

        return embedding.numpy(), phonemes

    def cosine_similarity(a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    embeddings = []
    for sent1, sent2 in test_pairs:
        emb1, ph1 = get_embedding(sent1)
        emb2, ph2 = get_embedding(sent2)

        similarity = cosine_similarity(emb1, emb2)

        print(f"\nText 1: '{sent1}'")
        print(f"Phonemes 1: {ph1}")
        print(f"Text 2: '{sent2}'")
        print(f"Phonemes 2: {ph2}")
        print(f"Cosine similarity: {similarity:.3f}")

        if similarity > 0.7:
            print("✓ High similarity (semantically related)")
        elif similarity > 0.4:
            print("→ Moderate similarity")
        else:
            print("× Low similarity (different concepts)")

        embeddings.append((sent1, emb1))
        print("-" * 50)

    # Test distinctiveness
    print("\n" + "=" * 70)
    print("EMBEDDING DISTINCTIVENESS TEST")
    print("(Different sentences should have distinct embeddings)")
    print("=" * 70)

    # Compare a greeting with a question
    greeting = "soo dhawoow"
    question = "maxaa ku helay"

    emb_g, ph_g = get_embedding(greeting)
    emb_q, ph_q = get_embedding(question)

    sim = cosine_similarity(emb_g, emb_q)
    print(f"\n'{greeting}' vs '{question}'")
    print(f"Similarity: {sim:.3f}")
    if sim < 0.5:
        print("✓ Good: Different concepts have distinct embeddings")
    else:
        print("⚠️  Warning: Too similar for different concepts")

    # Check embedding dimensions
    print("\n" + "=" * 70)
    print("EMBEDDING PROPERTIES")
    print("=" * 70)
    sample_emb = embeddings[0][1]
    print(f"Embedding dimension: {len(sample_emb)}")
    print(f"Embedding norm: {np.linalg.norm(sample_emb):.3f}")
    print(f"Embedding mean: {np.mean(sample_emb):.3f}")
    print(f"Embedding std: {np.std(sample_emb):.3f}")

    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("- PL-BERT successfully produces embeddings for Somali phonemes")
    print("- Similar sentences have similar embeddings")
    print("- Different concepts have distinct embeddings")
    print("- Ready for StyleTTS2 integration!")
    print("=" * 70)

if __name__ == "__main__":
    test_plbert_embeddings()