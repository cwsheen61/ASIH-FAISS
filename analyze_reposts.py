#!/usr/bin/env python3
"""
Analyze how often images are reposted in our dataset.
This tests if hateful content gets copied across threads.
"""
import json
import sqlite3
import numpy as np
from collections import defaultdict
from typing import List, Tuple
import config

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def main():
    print("="*70)
    print("REPOST ANALYSIS - 'Do hateful images get copied?'")
    print("="*70)

    db = sqlite3.connect(config.DB_PATH)

    # Get all labeled images
    cursor = db.execute("""
        SELECT id, post_id, clip_embedding, gemini_label, image_phash
        FROM posts
        WHERE clip_embedding IS NOT NULL
        AND gemini_label IS NOT NULL
        AND quarantined = 0
    """)
    all_images = cursor.fetchall()
    db.close()

    print(f"\n✓ Loaded {len(all_images)} labeled images")

    # Check for exact pHash duplicates
    phash_groups = defaultdict(list)
    for db_id, post_id, clip_emb, label, phash in all_images:
        if phash:
            phash_groups[phash].append((post_id, label))

    # Find pHash duplicates
    exact_duplicates = {k: v for k, v in phash_groups.items() if len(v) > 1}

    print(f"\nExact duplicates (same pHash):")
    print(f"  Total pHash groups with duplicates: {len(exact_duplicates)}")

    denied_reposts = 0
    allowed_reposts = 0

    for phash, posts in exact_duplicates.items():
        labels = [label for _, label in posts]
        if 'DENIED' in labels:
            denied_reposts += len(posts) - 1
        else:
            allowed_reposts += len(posts) - 1

    print(f"  DENIED images reposted: {denied_reposts} times")
    print(f"  ALLOWED images reposted: {allowed_reposts} times")

    # Now check for near-duplicates using CLIP similarity
    print(f"\n\nChecking for near-duplicates (CLIP similarity ≥0.98)...")

    near_duplicates = []
    checked = 0

    for i, (id1, post1, emb1, label1, phash1) in enumerate(all_images):
        if i % 100 == 0:
            print(f"  Checked {i}/{len(all_images)} images...", end='\r')

        vec1 = np.array(json.loads(emb1))

        for id2, post2, emb2, label2, phash2 in all_images[i+1:]:
            if phash1 == phash2:  # Skip exact duplicates we already counted
                continue

            vec2 = np.array(json.loads(emb2))
            sim = cosine_similarity(vec1, vec2)

            if sim >= 0.98:  # Very high similarity, likely same content
                near_duplicates.append({
                    'post1': post1,
                    'post2': post2,
                    'label1': label1,
                    'label2': label2,
                    'similarity': sim
                })
                checked += 1

    print(f"  Checked {len(all_images)}/{len(all_images)} images...    ")
    print(f"\n✓ Found {len(near_duplicates)} near-duplicate pairs (≥98% similar)")

    denied_near_reposts = sum(1 for d in near_duplicates if 'DENIED' in [d['label1'], d['label2']])
    allowed_near_reposts = len(near_duplicates) - denied_near_reposts

    print(f"  DENIED near-duplicates: {denied_near_reposts}")
    print(f"  ALLOWED near-duplicates: {allowed_near_reposts}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    total_reposts = denied_reposts + allowed_reposts + len(near_duplicates)
    total_denied_reposts = denied_reposts + denied_near_reposts

    print(f"\nTotal reposted content detected: {total_reposts}")
    print(f"  Hateful (DENIED) reposts: {total_denied_reposts}")
    print(f"  Clean (ALLOWED) reposts: {allowed_reposts + allowed_near_reposts}")

    if total_denied_reposts > 0:
        print(f"\n✓ YES - Hateful content IS being reposted!")
        print(f"  We could auto-deny {total_denied_reposts} reposts using similarity lookup")
    else:
        print(f"\n✗ NO - No hateful reposts detected in this dataset")

    # Show some examples
    if denied_reposts > 0:
        print(f"\nExample exact duplicates (DENIED):")
        count = 0
        for phash, posts in exact_duplicates.items():
            labels = [label for _, label in posts]
            if 'DENIED' in labels:
                post_ids = [post_id for post_id, _ in posts]
                print(f"  Posts {post_ids} - same image, DENIED")
                count += 1
                if count >= 3:
                    break

    print("="*70)

if __name__ == "__main__":
    main()
