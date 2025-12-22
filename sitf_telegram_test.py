#!/usr/bin/env python3
"""
SITF Test: Check if Telegram images match 4chan labeled content.
"Saw It There First" - detect cross-platform reposts.
"""
import json
import sqlite3
import numpy as np
from typing import List, Tuple
import config

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_similar_in_dataset(
    new_embedding: str,
    dataset: List[Tuple],
    threshold: float = 0.90
) -> List[Tuple]:
    """
    Find similar images in labeled dataset.

    Args:
        new_embedding: CLIP embedding JSON string
        dataset: List of (id, post_id, embedding, label, phash, reason) tuples
        threshold: Minimum similarity threshold

    Returns:
        List of (similarity, post_id, label, phash, reason) tuples above threshold
    """
    new_vec = np.array(json.loads(new_embedding))

    matches = []
    for db_id, post_id, emb, label, phash, reason in dataset:
        if not emb:
            continue

        vec = np.array(json.loads(emb))
        sim = cosine_similarity(new_vec, vec)

        if sim >= threshold:
            matches.append((sim, post_id, label, phash, reason))

    # Sort by similarity (highest first)
    matches.sort(reverse=True, key=lambda x: x[0])
    return matches

def main():
    print("="*70)
    print("SITF TEST - Saw It There First")
    print("Checking if Telegram images match 4chan labeled content")
    print("="*70)

    db = sqlite3.connect(config.DB_PATH)

    # Get 4chan labeled dataset
    print("\nLoading 4chan labeled dataset...")
    cursor = db.execute("""
        SELECT id, post_id, clip_embedding, gemini_label, image_phash, gemini_reason
        FROM posts
        WHERE clip_embedding IS NOT NULL
        AND gemini_label IS NOT NULL
        AND source LIKE '4chan/%'
        AND quarantined = 0
    """)
    chan_dataset = cursor.fetchall()
    print(f"✓ Loaded {len(chan_dataset)} labeled 4chan images")

    # Get recent Telegram images (unlabeled)
    cursor = db.execute("""
        SELECT id, post_id, clip_embedding, content_text, image_phash
        FROM posts
        WHERE clip_embedding IS NOT NULL
        AND source LIKE 'telegram/%'
        AND gemini_label IS NULL
        AND quarantined = 0
        ORDER BY id DESC
        LIMIT 30
    """)
    telegram_images = cursor.fetchall()
    db.close()

    print(f"✓ Loaded {len(telegram_images)} recent Telegram images")

    print("\n" + "="*70)
    print("SITF SIMILARITY RESULTS")
    print("="*70)

    # Thresholds
    HIGH_SIMILARITY = 0.95  # Almost identical
    MEDIUM_SIMILARITY = 0.90  # Very similar

    results = {
        'high_match': [],      # ≥0.95
        'medium_match': [],    # 0.90-0.95
        'no_match': [],        # <0.90
    }

    for tg_id, tg_post_id, tg_emb, tg_text, tg_phash in telegram_images:
        matches = find_similar_in_dataset(tg_emb, chan_dataset, threshold=MEDIUM_SIMILARITY)

        if not matches:
            results['no_match'].append(tg_post_id)
            print(f"\n? Telegram post {tg_post_id}: No match (novel content)")
            continue

        # Get best match
        similarity, chan_post_id, chan_label, chan_phash, chan_reason = matches[0]

        # Check if exact duplicate (same pHash)
        exact_dup = (tg_phash == chan_phash) if tg_phash and chan_phash else False

        if similarity >= HIGH_SIMILARITY:
            confidence = "HIGH"
            emoji = "✓✓"
            results['high_match'].append({
                'tg_post': tg_post_id,
                'chan_post': chan_post_id,
                'similarity': similarity,
                'label': chan_label,
                'exact_dup': exact_dup,
                'reason': chan_reason
            })
        else:  # MEDIUM_SIMILARITY
            confidence = "MEDIUM"
            emoji = "✓"
            results['medium_match'].append({
                'tg_post': tg_post_id,
                'chan_post': chan_post_id,
                'similarity': similarity,
                'label': chan_label,
                'exact_dup': exact_dup,
                'reason': chan_reason
            })

        dup_marker = " [EXACT DUPLICATE]" if exact_dup else ""
        print(f"\n{emoji} Telegram post {tg_post_id}{dup_marker}:")
        print(f"   Similarity: {similarity:.3f} ({confidence})")
        print(f"   Matches 4chan post: {chan_post_id}")
        print(f"   Can reuse label: {chan_label}")
        if similarity >= HIGH_SIMILARITY and chan_reason:
            print(f"   Reason: {chan_reason[:80]}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    total = len(telegram_images)
    high = len(results['high_match'])
    medium = len(results['medium_match'])
    no_match = len(results['no_match'])

    print(f"\nTotal Telegram images checked: {total}")
    print(f"\nSITF Match Distribution:")
    print(f"  HIGH (≥0.95):       {high} ({high/total*100:.1f}%)")
    print(f"    → We saw this on 4chan! Auto-moderate with high confidence")
    print(f"  MEDIUM (0.90-0.95): {medium} ({medium/total*100:.1f}%)")
    print(f"    → Probably saw this on 4chan, use with caution")
    print(f"  NO MATCH (<0.90):   {no_match} ({no_match/total*100:.1f}%)")
    print(f"    → Novel to Telegram, needs fresh moderation")

    # Count exact duplicates
    exact_dups_high = sum(1 for m in results['high_match'] if m['exact_dup'])
    exact_dups_med = sum(1 for m in results['medium_match'] if m['exact_dup'])

    print(f"\nExact duplicates (same pHash): {exact_dups_high + exact_dups_med}")
    print(f"  → Would have been caught by pHash alone")

    near_dups_high = high - exact_dups_high
    near_dups_med = medium - exact_dups_med

    print(f"\nNear-duplicates (CLIP only): {near_dups_high + near_dups_med}")
    print(f"  → CLIP similarity adds value here!")

    print("\n" + "="*70)
    print("SITF EFFECTIVENESS")
    print("="*70)

    sitf_hits = high + medium
    print(f"\n{sitf_hits}/{total} Telegram images ({sitf_hits/total*100:.1f}%) match 4chan content")

    if sitf_hits > 0:
        print(f"\n✓ SUCCESS - SITF is working!")
        print(f"  Content IS being reposted from 4chan to Telegram")
        print(f"  We can auto-moderate {sitf_hits} images using cached labels")

        # Count DENIED matches
        denied_matches = [m for m in results['high_match'] + results['medium_match'] if m['label'] == 'DENIED']
        if denied_matches:
            print(f"\n⚠ {len(denied_matches)} hateful images from 4chan appeared on Telegram:")
            for match in denied_matches[:3]:
                print(f"    4chan {match['chan_post']} → Telegram {match['tg_post']} (sim: {match['similarity']:.3f})")
    else:
        print(f"\n✗ No matches found")
        print(f"  Either:")
        print(f"  1. Telegram and 4chan don't share content")
        print(f"  2. We need more labeled data")
        print(f"  3. The content is genuinely different")

    print("="*70)

if __name__ == "__main__":
    main()
