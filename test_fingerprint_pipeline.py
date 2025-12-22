#!/usr/bin/env python3
"""
Test the complete fingerprint-based moderation pipeline.
Demonstrates SITF (Saw It There First) and ASIH (Already Saw It Here) strategy.
"""
import sqlite3
from fingerprint_lookup import FingerprintLookup
import config

def test_pipeline():
    print("="*70)
    print("FINGERPRINT-BASED MODERATION PIPELINE TEST")
    print("="*70)

    lookup = FingerprintLookup()

    # Get some test images with labels
    db = sqlite3.connect(config.DB_PATH)

    # Test 1: Exact image duplicate (pHash)
    print("\n" + "="*70)
    print("TEST 1: Exact Image Duplicate Detection (pHash)")
    print("="*70)

    cursor = db.execute("""
        SELECT image_phash, gemini_label, post_id
        FROM posts
        WHERE image_phash IS NOT NULL
        AND gemini_label IS NOT NULL
        AND quarantined = 0
        LIMIT 1
    """)
    row = cursor.fetchone()

    if row:
        test_phash, actual_label, post_id = row
        print(f"\nTest post: {post_id}")
        print(f"Actual label: {actual_label}")
        print(f"pHash: {test_phash[:16]}...")

        result = lookup.find_similar_content(image_phash=test_phash)

        if result:
            label, reason, match_type, confidence = result
            print(f"\n✓ MATCH FOUND via {match_type}")
            print(f"  Predicted label: {label}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Correct: {label == actual_label}")
        else:
            print("\n✗ No match (unexpected)")

    # Test 2: Similar image (CLIP)
    print("\n" + "="*70)
    print("TEST 2: Similar Image Detection (CLIP ≥0.90)")
    print("="*70)

    cursor = db.execute("""
        SELECT clip_embedding, gemini_label, post_id
        FROM posts
        WHERE clip_embedding IS NOT NULL
        AND gemini_label IS NOT NULL
        AND quarantined = 0
        LIMIT 1
    """)
    row = cursor.fetchone()

    if row:
        test_clip, actual_label, post_id = row
        print(f"\nTest post: {post_id}")
        print(f"Actual label: {actual_label}")

        result = lookup.find_similar_content(
            clip_embedding=test_clip,
            image_similarity_threshold=0.90
        )

        if result:
            label, reason, match_type, confidence = result
            print(f"\n✓ MATCH FOUND via {match_type}")
            print(f"  Predicted label: {label}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Correct: {label == actual_label}")
        else:
            print("\n✗ No match")

    # Test 3: Text fingerprints
    print("\n" + "="*70)
    print("TEST 3: Text Fingerprint Detection (New Feature)")
    print("="*70)

    cursor = db.execute("""
        SELECT text_hash, text_embedding, gemini_label, post_id
        FROM posts
        WHERE text_hash IS NOT NULL
        AND text_hash != ''
        AND gemini_label IS NOT NULL
        AND quarantined = 0
        LIMIT 1
    """)
    row = cursor.fetchone()

    if row:
        test_text_hash, test_text_emb, actual_label, post_id = row
        print(f"\nTest post: {post_id}")
        print(f"Actual label: {actual_label}")
        print(f"Text hash: {test_text_hash[:16]}...")

        result = lookup.find_similar_content(
            text_hash=test_text_hash,
            text_embedding=test_text_emb
        )

        if result:
            label, reason, match_type, confidence = result
            print(f"\n✓ MATCH FOUND via {match_type}")
            print(f"  Predicted label: {label}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Correct: {label == actual_label}")
        else:
            print("\n✗ No match")
    else:
        print("\nNo posts with text fingerprints yet")
        print("(Need to scrape with updated pipeline)")

    # Statistics
    print("\n" + "="*70)
    print("DATABASE STATISTICS")
    print("="*70)

    stats = {}
    stats['total'] = db.execute("SELECT COUNT(*) FROM posts WHERE quarantined = 0").fetchone()[0]
    stats['labeled'] = db.execute("SELECT COUNT(*) FROM posts WHERE gemini_label IS NOT NULL AND quarantined = 0").fetchone()[0]
    stats['with_phash'] = db.execute("SELECT COUNT(*) FROM posts WHERE image_phash IS NOT NULL AND quarantined = 0").fetchone()[0]
    stats['with_clip'] = db.execute("SELECT COUNT(*) FROM posts WHERE clip_embedding IS NOT NULL AND quarantined = 0").fetchone()[0]
    stats['with_ocr'] = db.execute("SELECT COUNT(*) FROM posts WHERE ocr_text IS NOT NULL AND ocr_text != '' AND quarantined = 0").fetchone()[0]
    stats['with_text_hash'] = db.execute("SELECT COUNT(*) FROM posts WHERE text_hash IS NOT NULL AND text_hash != '' AND quarantined = 0").fetchone()[0]

    print(f"\nTotal posts (active): {stats['total']}")
    print(f"Labeled posts: {stats['labeled']}")
    print(f"\nFingerprint coverage:")
    print(f"  Image pHash: {stats['with_phash']} ({stats['with_phash']/stats['total']*100 if stats['total'] > 0 else 0:.1f}%)")
    print(f"  CLIP embeddings: {stats['with_clip']} ({stats['with_clip']/stats['total']*100 if stats['total'] > 0 else 0:.1f}%)")
    print(f"  OCR text: {stats['with_ocr']} ({stats['with_ocr']/stats['total']*100 if stats['total'] > 0 else 0:.1f}%)")
    print(f"  Text fingerprints: {stats['with_text_hash']} ({stats['with_text_hash']/stats['total']*100 if stats['total'] > 0 else 0:.1f}%)")

    db.close()

    # Summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print("\nThe fingerprint-based moderation system:")
    print("  1. pHash - Catches exact image duplicates (fast)")
    print("  2. CLIP - Catches similar/edited images (semantic)")
    print("  3. Text hash - Catches exact text matches (fast)")
    print("  4. Text embedding - Catches similar text/rephrasing (semantic)")
    print("\nThis enables SITF/ASIH strategy:")
    print("  - SITF: Scrape hate sources, build fingerprint database")
    print("  - ASIH: Reuse moderation decisions via fingerprints")
    print("  - No images/text stored - only encrypted fingerprints")
    print("="*70)

if __name__ == "__main__":
    test_pipeline()
