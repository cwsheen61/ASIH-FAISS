#!/usr/bin/env python3
"""
Check how many new images have similar matches in our labeled dataset.
This tests the "I've seen this before" concept.
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
    """
    Find the most similar labeled images to a new image.

    Args:
        new_embedding: CLIP embedding JSON string
        labeled_embeddings: List of (id, post_id, embedding, label, reason) tuples
        top_k: Number of top matches to return

    Returns:
        List of (similarity, post_id, label, reason) tuples
    """
    new_vec = np.array(json.loads(new_embedding))

    similarities = []
    for db_id, post_id, emb, label, reason in labeled_embeddings:
        if not emb:
            continue
        vec = np.array(json.loads(emb))
        sim = cosine_similarity(new_vec, vec)
        similarities.append((sim, post_id, label, reason))

    # Sort by similarity (highest first)
    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:top_k]

def main():
    print("="*70)
    print("CLIP SIMILARITY LOOKUP - 'Have we seen this before?'")
    print("="*70)

    db = sqlite3.connect(config.DB_PATH)

    # Get all labeled images with CLIP embeddings
    print("\nLoading labeled dataset...")
    cursor = db.execute("""
        SELECT id, post_id, clip_embedding, gemini_label, gemini_reason
        FROM posts
        WHERE clip_embedding IS NOT NULL
        AND gemini_label IS NOT NULL
        AND quarantined = 0
    """)
    labeled_data = cursor.fetchall()
    print(f"✓ Loaded {len(labeled_data)} labeled images")

    # Get the 41 new unlabeled images
    cursor = db.execute("""
        SELECT id, post_id, clip_embedding, content_text, gemini_label
        FROM posts
        WHERE clip_embedding IS NOT NULL
        AND quarantined = 0
        ORDER BY id DESC
        LIMIT 41
    """)
    new_images = cursor.fetchall()
    db.close()

    print(f"✓ Loaded {len(new_images)} new images to check\n")
    print("="*70)
    print("SIMILARITY SEARCH RESULTS")
    print("="*70)

    # Similarity thresholds
    HIGH_SIMILARITY = 0.95  # Very similar
    MEDIUM_SIMILARITY = 0.85  # Somewhat similar

    results = {
        'high_confidence': [],  # >0.95 similarity
        'medium_confidence': [],  # 0.85-0.95 similarity
        'no_match': [],  # <0.85 similarity
    }

    for db_id, post_id, clip_emb, text, actual_label in new_images:
        # Find most similar labeled image
        matches = find_similar_images(clip_emb, labeled_data, top_k=1)

        if not matches:
            print(f"\n❌ Post {post_id}: No matches found")
            continue

        similarity, matched_post_id, matched_label, matched_reason = matches[0]

        # Classify confidence level
        if similarity >= HIGH_SIMILARITY:
            confidence = "HIGH"
            emoji = "✓✓"
            results['high_confidence'].append({
                'post_id': post_id,
                'similarity': similarity,
                'matched_label': matched_label,
                'actual_label': actual_label,
                'correct': matched_label == actual_label if actual_label else None
            })
        elif similarity >= MEDIUM_SIMILARITY:
            confidence = "MEDIUM"
            emoji = "✓"
            results['medium_confidence'].append({
                'post_id': post_id,
                'similarity': similarity,
                'matched_label': matched_label,
                'actual_label': actual_label,
                'correct': matched_label == actual_label if actual_label else None
            })
        else:
            confidence = "LOW"
            emoji = "?"
            results['no_match'].append({
                'post_id': post_id,
                'similarity': similarity,
                'matched_label': matched_label,
                'actual_label': actual_label,
                'correct': matched_label == actual_label if actual_label else None
            })

        print(f"\n{emoji} Post {post_id}:")
        print(f"   Similarity: {similarity:.3f} ({confidence} confidence)")
        print(f"   Matched to: Post {matched_post_id}")
        print(f"   Predicted label: {matched_label}")
        if actual_label:
            match_emoji = "✓" if matched_label == actual_label else "✗"
            print(f"   Actual label: {actual_label} {match_emoji}")
            print(f"   Reason: {matched_reason[:80]}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    total = len(new_images)
    high = len(results['high_confidence'])
    medium = len(results['medium_confidence'])
    low = len(results['no_match'])

    print(f"\nTotal images checked: {total}")
    print(f"\nSimilarity Distribution:")
    print(f"  HIGH (≥0.95):     {high} ({high/total*100:.1f}%) - 'Definitely seen before'")
    print(f"  MEDIUM (0.85-0.95): {medium} ({medium/total*100:.1f}%) - 'Probably seen similar'")
    print(f"  LOW (<0.85):      {low} ({low/total*100:.1f}%) - 'Novel/unseen'")

    # Accuracy for high confidence matches
    high_with_labels = [r for r in results['high_confidence'] if r['actual_label']]
    if high_with_labels:
        high_correct = sum(1 for r in high_with_labels if r['correct'])
        print(f"\nHigh Confidence Accuracy: {high_correct}/{len(high_with_labels)} ({high_correct/len(high_with_labels)*100:.1f}%)")

    # Accuracy for medium confidence matches
    medium_with_labels = [r for r in results['medium_confidence'] if r['actual_label']]
    if medium_with_labels:
        medium_correct = sum(1 for r in medium_with_labels if r['correct'])
        print(f"Medium Confidence Accuracy: {medium_correct}/{len(medium_with_labels)} ({medium_correct/len(medium_with_labels)*100:.1f}%)")

    print("\n" + "="*70)
    print("CONCLUSION:")
    print(f"  {high} images have high-similarity matches (can use cached labels)")
    print(f"  {medium + low} images are novel/different (need fresh moderation)")
    print("="*70)

if __name__ == "__main__":
    main()
