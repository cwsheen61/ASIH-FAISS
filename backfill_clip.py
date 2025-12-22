#!/usr/bin/env python3
"""Backfill CLIP embeddings for existing images."""
import base64
import sqlite3
import time
from pathlib import Path
import config
from clip_encoder import generate_clip_embedding


def backfill_clip_embeddings():
    """Generate CLIP embeddings for all images that don't have one."""
    conn = sqlite3.connect(config.DB_PATH)

    # Find images without CLIP embeddings
    cursor = conn.execute("""
        SELECT id, encoded_path
        FROM posts
        WHERE encoded_path IS NOT NULL
        AND quarantined = 0
        AND clip_embedding IS NULL
    """)

    posts = cursor.fetchall()
    total = len(posts)

    if total == 0:
        print("No images need CLIP embeddings!")
        return

    print(f"{'='*70}")
    print(f"CLIP EMBEDDING BACKFILL")
    print(f"{'='*70}")
    print(f"Found {total} images to process\n")

    start_time = time.time()
    success_count = 0
    fail_count = 0

    for i, (post_id, encoded_path) in enumerate(posts, 1):
        print(f"[{i}/{total}] Processing post {post_id}...", end=" ")

        try:
            # Load the encoded image
            with open(encoded_path, 'rb') as f:
                encoded_data = f.read()
            image_data = base64.b64decode(encoded_data)

            # Generate CLIP embedding
            clip_embedding = generate_clip_embedding(image_data)

            if clip_embedding:
                # Update database
                conn.execute(
                    "UPDATE posts SET clip_embedding = ? WHERE id = ?",
                    (clip_embedding, post_id)
                )
                conn.commit()
                print(f"✓")
                success_count += 1
            else:
                print(f"✗ FAILED (no embedding)")
                fail_count += 1

        except Exception as e:
            print(f"✗ ERROR: {e}")
            fail_count += 1

        # Progress update every 50 images
        if i % 50 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (total - i) / rate if rate > 0 else 0
            print(f"\n  Progress: {i}/{total} ({i/total*100:.1f}%)")
            print(f"  Rate: {rate:.2f} images/sec")
            print(f"  ETA: {remaining/60:.1f} minutes\n")

    conn.close()

    # Final summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"BACKFILL COMPLETE")
    print(f"{'='*70}")
    print(f"Total processed:  {total}")
    print(f"Success:          {success_count}")
    print(f"Failed:           {fail_count}")
    print(f"Time:             {elapsed:.1f} seconds")
    print(f"Rate:             {success_count/elapsed:.2f} images/sec")
    print(f"{'='*70}")


if __name__ == "__main__":
    backfill_clip_embeddings()
