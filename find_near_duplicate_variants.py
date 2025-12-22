#!/usr/bin/env python3
"""
Find near-duplicate variants: high CLIP similarity but different pHash.
This detects cropped/edited versions of the same content.
"""
import json
import sqlite3
import numpy as np
from typing import List
import config

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def main():
    print("="*70)
    print("NEAR-DUPLICATE VARIANT DETECTION")
    print("Finding images with high CLIP similarity but different pHash")
    print("="*70)

    db = sqlite3.connect(config.DB_PATH)

    # Get all labeled images with CLIP embeddings
    cursor = db.execute("""
        SELECT id, post_id, clip_embedding, gemini_label, image_phash
        FROM posts
        WHERE clip_embedding IS NOT NULL
        AND gemini_label IS NOT NULL
        AND quarantined = 0
    """)
    images = cursor.fetchall()
    db.close()

    print(f"\n✓ Loaded {len(images)} labeled images")
    print(f"Searching for near-duplicates (CLIP ≥0.95, different pHash)...")

    near_duplicates = []
    checked = 0

    # Compare all pairs
    for i, (id1, post1, emb1, label1, phash1) in enumerate(images):
        if i % 100 == 0 and i > 0:
            print(f"  Checked {i}/{len(images)} images...", end='\r')

        vec1 = np.array(json.loads(emb1))

        for id2, post2, emb2, label2, phash2 in images[i+1:]:
            # Skip if same pHash (exact duplicate)
            if phash1 == phash2:
                continue

            vec2 = np.array(json.loads(emb2))
            sim = cosine_similarity(vec1, vec2)

            if sim >= 0.95:  # High similarity, likely variant
                near_duplicates.append({
                    'post1': post1,
                    'post2': post2,
                    'label1': label1,
                    'label2': label2,
                    'similarity': sim,
                    'phash1': phash1,
                    'phash2': phash2
                })

    print(f"  Checked {len(images)}/{len(images)} images...    ")
    print(f"\n✓ Found {len(near_duplicates)} near-duplicate pairs")

    # Categorize by labels
    denied_variants = [d for d in near_duplicates if d['label1'] == 'DENIED' and d['label2'] == 'DENIED']
    allowed_variants = [d for d in near_duplicates if d['label1'] == 'ALLOWED' and d['label2'] == 'ALLOWED']
    mixed_variants = [d for d in near_duplicates if d['label1'] != d['label2']]

    print("\n" + "="*70)
    print("BREAKDOWN")
    print("="*70)

    print(f"\nBoth DENIED (same hateful content, different version): {len(denied_variants)}")
    print(f"Both ALLOWED (same clean content, different version): {len(allowed_variants)}")
    print(f"Mixed labels (high similarity but different moderation): {len(mixed_variants)}")

    # Summary
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    if len(denied_variants) > 0:
        print(f"\n✓ YES - We found {len(denied_variants)} pairs of DENIED content variants!")
        print(f"  These are likely cropped/edited/re-compressed versions of the same hateful meme")
        print(f"\nCLIP similarity CAN help catch these variants that pHash misses!")
        print(f"\nPotential workflow:")
        print(f"  1. pHash catches exact duplicates ({18} caught)")
        print(f"  2. CLIP similarity catches near-duplicates ({len(denied_variants)} pairs found)")
        print(f"  3. Text analysis confirms final decision")
    else:
        print(f"\n✗ No DENIED variants detected with CLIP ≥0.95")
        print(f"  This suggests:")
        print(f"  1. Hateful memes aren't being edited/cropped before reposting")
        print(f"  2. OR we need more labeled data to find variants")
        print(f"  3. OR variants are less similar than 0.95 threshold")

    # Show examples
    if denied_variants:
        print(f"\nExamples of DENIED variants (likely same meme, edited):")
        for variant in denied_variants[:5]:
            print(f"  Post {variant['post1']} ≈ Post {variant['post2']}")
            print(f"    Similarity: {variant['similarity']:.3f}, Different pHash")

    if mixed_variants:
        print(f"\nWarning: {len(mixed_variants)} high-similarity pairs have DIFFERENT labels")
        print("These might be:")
        print("  - Labeling inconsistencies")
        print("  - Similar but not identical content")
        for variant in mixed_variants[:3]:
            print(f"  Post {variant['post1']} ({variant['label1']}) ≈ Post {variant['post2']} ({variant['label2']})")
            print(f"    Similarity: {variant['similarity']:.3f}")

    print("="*70)

if __name__ == "__main__":
    main()
