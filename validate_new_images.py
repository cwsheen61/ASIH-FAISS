#!/usr/bin/env python3
"""Label new images with Gemini and compare with CLIP classifier predictions."""
import json
import sqlite3
from clip_classifier import CLIPClassifier
from gemini_labeler import GeminiLabeler
import config
import time

def main():
    print("="*70)
    print("VALIDATING CLIP CLASSIFIER ON NEW IMAGES")
    print("="*70)

    # Load trained CLIP classifier
    clip_classifier = CLIPClassifier()
    print("✓ Loaded CLIP classifier")

    # Initialize Gemini labeler
    gemini_labeler = GeminiLabeler()
    print("✓ Loaded Gemini labeler\n")

    # Get recent unlabeled posts with CLIP embeddings (our 35 new images)
    db = sqlite3.connect(config.DB_PATH)
    cursor = db.execute("""
        SELECT id, post_id, source, clip_embedding, encoded_path, content_text, has_image, image_phash
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

    print(f"Found {len(new_images)} new images to validate\n")
    print("="*70)
    print("LABELING WITH GEMINI AND COMPARING...")
    print("="*70)

    results = []
    correct = 0
    total = 0

    for db_id, post_id, source, clip_emb, encoded_path, text, has_image, phash in new_images:
        # Get CLIP prediction
        clip_label, clip_conf = clip_classifier.predict(clip_emb)

        # Get Gemini label
        print(f"\nPost {post_id}:")
        print(f"  CLIP predicted: {clip_label} ({clip_conf:.2%})")

        # Check for duplicate label
        gemini_label = None
        gemini_reason = None

        if phash:
            db = sqlite3.connect(config.DB_PATH)
            cursor = db.execute("""
                SELECT gemini_label, gemini_reason
                FROM posts
                WHERE image_phash = ? AND gemini_label IS NOT NULL
                LIMIT 1
            """, (phash,))
            existing = cursor.fetchone()
            db.close()

            if existing:
                gemini_label, gemini_reason = existing
                print(f"  Gemini: {gemini_label} (from duplicate) - {gemini_reason[:80]}")

        if not gemini_label:
            # Call Gemini API
            image_path = encoded_path if has_image else None
            result = gemini_labeler.classify_content(text, image_path)

            if result:
                gemini_label, gemini_reason = result
                print(f"  Gemini: {gemini_label} - {gemini_reason[:80]}")

                # Save to database
                db = sqlite3.connect(config.DB_PATH)
                db.execute("""
                    UPDATE posts
                    SET gemini_label = ?, gemini_reason = ?
                    WHERE id = ?
                """, (gemini_label, gemini_reason, db_id))
                db.commit()
                db.close()
            else:
                print(f"  Gemini: FAILED")
                continue

            time.sleep(1.0)  # Rate limiting

        # Compare
        match = clip_label == gemini_label
        total += 1
        if match:
            correct += 1
            print(f"  Result: ✓ MATCH")
        else:
            print(f"  Result: ✗ MISMATCH")

        results.append({
            'post_id': post_id,
            'clip_label': clip_label,
            'clip_conf': clip_conf,
            'gemini_label': gemini_label,
            'gemini_reason': gemini_reason,
            'match': match,
            'text': text[:100] if text else ""
        })

    # Calculate metrics
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)

    accuracy = correct / total if total > 0 else 0
    print(f"Total validated: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print()

    # Breakdown by label
    clip_allowed = sum(1 for r in results if r['clip_label'] == 'ALLOWED')
    clip_denied = sum(1 for r in results if r['clip_label'] == 'DENIED')
    gemini_allowed = sum(1 for r in results if r['gemini_label'] == 'ALLOWED')
    gemini_denied = sum(1 for r in results if r['gemini_label'] == 'DENIED')

    print("Label Distribution:")
    print(f"  CLIP:   {clip_allowed} ALLOWED, {clip_denied} DENIED")
    print(f"  Gemini: {gemini_allowed} ALLOWED, {gemini_denied} DENIED")
    print()

    # Show mismatches
    mismatches = [r for r in results if not r['match']]
    if mismatches:
        print(f"Mismatches ({len(mismatches)}):")
        for r in mismatches:
            print(f"\n  Post {r['post_id']}:")
            print(f"    CLIP: {r['clip_label']} ({r['clip_conf']:.2%})")
            print(f"    Gemini: {r['gemini_label']}")
            print(f"    Reason: {r['gemini_reason'][:80]}")
            if r['text']:
                print(f"    Text: {r['text'][:80]}")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
