#!/usr/bin/env python3
"""Classify newly scraped images using the trained CLIP classifier."""
import json
import sqlite3
from pathlib import Path
from clip_classifier import CLIPClassifier
import config

def main():
    print("="*70)
    print("CLASSIFYING NEW IMAGES WITH CLIP CLASSIFIER")
    print("="*70)

    # Load trained classifier
    classifier = CLIPClassifier()
    print("✓ Loaded trained CLIP classifier\n")

    # Get recent unlabeled posts with CLIP embeddings
    db = sqlite3.connect(config.DB_PATH)
    cursor = db.execute("""
        SELECT id, post_id, source, clip_embedding, encoded_path, content_text
        FROM posts
        WHERE clip_embedding IS NOT NULL
        AND gemini_label IS NULL
        AND quarantined = 0
        ORDER BY id DESC
        LIMIT 35
    """)

    new_images = cursor.fetchall()
    db.close()

    if not new_images:
        print("No new unlabeled images found")
        return

    print(f"Found {len(new_images)} new unlabeled images\n")
    print("="*70)
    print("PREDICTIONS:")
    print("="*70)

    # Classify each image
    results = {
        'ALLOWED': [],
        'DENIED': []
    }

    for db_id, post_id, source, clip_emb, encoded_path, text in new_images:
        label, confidence = classifier.predict(clip_emb)
        results[label].append((post_id, confidence, source, text[:80] if text else ""))

        # Show prediction
        emoji = "✓" if label == "ALLOWED" else "⚠"
        print(f"{emoji} Post {post_id} [{source}]")
        print(f"   Prediction: {label} (confidence: {confidence:.2%})")
        if text:
            clean_text = text[:100].replace('\n', ' ')
            print(f"   Text: {clean_text}...")
        print()

    # Summary
    print("="*70)
    print("SUMMARY:")
    print("="*70)
    print(f"Total classified: {len(new_images)}")
    print(f"  ALLOWED: {len(results['ALLOWED'])} ({len(results['ALLOWED'])/len(new_images)*100:.1f}%)")
    print(f"  DENIED:  {len(results['DENIED'])} ({len(results['DENIED'])/len(new_images)*100:.1f}%)")
    print()

    # Show top confidence predictions
    print("Top 5 ALLOWED (by confidence):")
    for post_id, conf, source, text in sorted(results['ALLOWED'], key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Post {post_id}: {conf:.2%} - {text[:60]}")

    print("\nTop 5 DENIED (by confidence):")
    for post_id, conf, source, text in sorted(results['DENIED'], key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Post {post_id}: {conf:.2%} - {text[:60]}")

    print("="*70)

if __name__ == "__main__":
    main()
