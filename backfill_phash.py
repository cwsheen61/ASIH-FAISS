#!/usr/bin/env python3
"""Backfill phash values for existing images."""
import base64
from io import BytesIO
from PIL import Image
import imagehash
import sqlite3
import config


def calculate_phash(encoded_path: str) -> str:
    """Calculate perceptual hash from encoded image."""
    try:
        with open(encoded_path, 'rb') as f:
            encoded_data = f.read()
        image_data = base64.b64decode(encoded_data)
        image = Image.open(BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        phash = imagehash.phash(image)
        return str(phash)
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    """Backfill phash for all images."""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.execute("""
        SELECT id, encoded_path
        FROM posts
        WHERE has_image = 1 AND image_phash IS NULL AND encoded_path IS NOT NULL
    """)

    posts = cursor.fetchall()
    print(f"Found {len(posts)} images to process")

    for post_id, encoded_path in posts:
        print(f"Processing post {post_id}...", end=" ")
        phash = calculate_phash(encoded_path)
        if phash:
            conn.execute(
                "UPDATE posts SET image_phash = ? WHERE id = ?",
                (phash, post_id)
            )
            print(f"✓ {phash}")
        else:
            print("✗ FAILED")

    conn.commit()
    conn.close()
    print("\nBackfill complete!")


if __name__ == "__main__":
    main()
