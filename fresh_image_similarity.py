#!/usr/bin/env python3
"""
Check how many TRULY NEW images match our OLDER labeled dataset.
This answers: "Have we seen similar images before?"
"""
import json
import sqlite3
import numpy as np
from typing import List, Tuple
import config

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_similar_images(new_embedding: str, labeled_embeddings: List[Tuple], top_k: int = 1) -> List[Tuple]:
    """Find the most similar labeled images to a new image."""
    new_vec = np.array(json.loads(new_embedding))

    similarities = []
    for db_id, post_id, emb, label, reason in labeled_embeddings:
        if not emb:
            continue
        vec = np.array(json.loads(emb))
        sim = cosine_similarity(new_vec, vec)
        similarities.append((sim, post_id, label, reason))

    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:top_k]

def main():
    print("="*70)
    print("FRESH IMAGE SIMILARITY - 'Have we seen this before?'")
    print("="*70)

    db = sqlite3.connect(config.DB_PATH)

    # Get OLDER labeled images (before the recent scraping)
    # We'll use images with ID < 21493 (before today's scraping started)
    print("\nLoading OLDER labeled dataset (training set)...")
    cursor = db.execute("""
        SELECT id, post_id, clip_embedding, gemini_label, gemini_reason
        FROM posts
        WHERE clip_embedding IS NOT NULL
        AND gemini_label IS NOT NULL
        AND quarantined = 0
        AND id < 21493
    """)
    labeled_data = cursor.fetchall()
    print(f"✓ Loaded {len(labeled_data)} OLDER labeled images")

    # Get 30 NEWEST unlabeled images (from today's scraping)
    cursor = db.execute("""
        SELECT id, post_id, clip_embedding, content_text
        FROM posts
        WHERE clip_embedding IS NOT NULL
        AND gemini_label IS NULL
        AND quarantined = 0
        ORDER BY id DESC
        LIMIT 30
    """)
    new_images = cursor.fetchall()
    db.close()

    print(f"✓ Loaded {len(new_images)} brand new images to check\n")
    print("="*70)
    print("SIMILARITY SEARCH RESULTS")
    print("="*70)

    HIGH_SIMILARITY = 0.95  # Very similar
    MEDIUM_SIMILARITY = 0.85  # Somewhat similar

    results = {
        'high_confidence': [],
        'medium_confidence': [],
        'no_match': [],
    }

    for db_id, post_id, clip_emb, text in new_images:
        matches = find_similar_images(clip_emb, labeled_data, top_k=1)

        if not matches:
            print(f"\n❌ Post {post_id}: No matches found")
            continue

        similarity, matched_post_id, matched_label, matched_reason = matches[0]

        if similarity >= HIGH_SIMILARITY:
            confidence = "HIGH"
            emoji = "✓✓"
            results['high_confidence'].append({
                'post_id': post_id,
                'similarity': similarity,
                'matched_label': matched_label,
                'matched_post_id': matched_post_id
            })
        elif similarity >= MEDIUM_SIMILARITY:
            confidence = "MEDIUM"
            emoji = "✓"
            results['medium_confidence'].append({
                'post_id': post_id,
                'similarity': similarity,
                'matched_label': matched_label,
                'matched_post_id': matched_post_id
            })
        else:
            confidence = "LOW"
            emoji = "?"
            results['no_match'].append({
                'post_id': post_id,
                'similarity': similarity,
                'matched_label': matched_label,
                'matched_post_id': matched_post_id
            })

        print(f"\n{emoji} Post {post_id} (NEW):")
        print(f"   Similarity: {similarity:.3f} ({confidence} confidence)")
        print(f"   Best match: Post {matched_post_id} (OLD)")
        print(f"   Can reuse label: {matched_label}")
        if similarity >= HIGH_SIMILARITY:
            print(f"   Reason: {matched_reason[:80]}")
        if text:
            clean_text = text[:80].replace('<', '').replace('>', '').replace('\n', ' ')
            print(f"   Text: {clean_text}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    total = len(new_images)
    high = len(results['high_confidence'])
    medium = len(results['medium_confidence'])
    low = len(results['no_match'])

    print(f"\nTotal NEW images checked: {total}")
    print(f"\nSimilarity Distribution:")
    print(f"  HIGH (≥0.95):       {high} ({high/total*100:.1f}%)")
    print(f"    → Can reuse cached labels confidently")
    print(f"  MEDIUM (0.85-0.95): {medium} ({medium/total*100:.1f}%)")
    print(f"    → Somewhat similar, maybe reuse")
    print(f"  LOW (<0.85):        {low} ({low/total*100:.1f}%)")
    print(f"    → Novel/unseen, need fresh moderation")

    print("\n" + "="*70)
    print("CONCLUSION:")
    print(f"  {high}/{total} images ({high/total*100:.1f}%) can use cached labels")
    print(f"  {medium+low}/{total} images ({(medium+low)/total*100:.1f}%) need fresh moderation")
    print("="*70)

    # Show some high confidence matches
    if results['high_confidence']:
        print("\nHigh Confidence Matches (sample):")
        for item in results['high_confidence'][:5]:
            print(f"  New Post {item['post_id']} ≈ Old Post {item['matched_post_id']}")
            print(f"    Similarity: {item['similarity']:.3f}, Label: {item['matched_label']}")

if __name__ == "__main__":
    main()
