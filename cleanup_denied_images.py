#!/usr/bin/env python3
"""Clean up DENIED images - delete pixels, keep metadata for duplicate detection."""
import os
import sqlite3
from pathlib import Path
import config


def cleanup_denied_images(dry_run: bool = True):
    """Delete image files for DENIED posts while keeping metadata.

    This implements the privacy-preserving architecture:
    - DENIED posts: Keep only pHash + CLIP + label (no visual data)
    - ALLOWED posts: Keep full image for training

    Args:
        dry_run: If True, show what would be deleted without actually deleting
    """
    conn = sqlite3.connect(config.DB_PATH)

    # Find all DENIED posts with image files
    cursor = conn.execute("""
        SELECT id, post_id, encoded_path, image_phash, clip_embedding
        FROM posts
        WHERE gemini_label = 'DENIED'
        AND encoded_path IS NOT NULL
        AND quarantined = 0
    """)

    denied_posts = cursor.fetchall()
    total = len(denied_posts)

    if total == 0:
        print("No DENIED images found to clean up.")
        conn.close()
        return

    print(f"{'='*70}")
    print(f"DENIED IMAGE CLEANUP")
    print(f"{'='*70}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE - DELETING FILES'}")
    print(f"Found {total} DENIED images to process\n")

    deleted_count = 0
    kept_count = 0
    missing_count = 0
    error_count = 0

    for post_id, post_num, encoded_path, phash, clip_embedding in denied_posts:
        # Verify metadata exists
        if not phash or not clip_embedding:
            print(f"  [WARNING] Post {post_num} missing metadata (phash={bool(phash)}, clip={bool(clip_embedding)})")
            kept_count += 1
            continue

        # Check if file exists
        if not os.path.exists(encoded_path):
            missing_count += 1
            continue

        # Get file size for reporting
        try:
            file_size = os.path.getsize(encoded_path)

            if dry_run:
                print(f"  [DRY RUN] Would delete: {encoded_path} ({file_size:,} bytes)")
                deleted_count += 1
            else:
                # Actually delete the file
                os.remove(encoded_path)

                # Update database to mark file as deleted
                conn.execute("""
                    UPDATE posts
                    SET encoded_path = NULL
                    WHERE id = ?
                """, (post_id,))

                print(f"  [DELETED] {encoded_path} ({file_size:,} bytes)")
                deleted_count += 1

        except Exception as e:
            print(f"  [ERROR] Failed to delete {encoded_path}: {e}")
            error_count += 1

    if not dry_run:
        conn.commit()

    conn.close()

    # Summary
    print(f"\n{'='*70}")
    print(f"CLEANUP SUMMARY")
    print(f"{'='*70}")
    print(f"Total DENIED posts:   {total}")
    print(f"{'Would delete' if dry_run else 'Deleted'}:            {deleted_count}")
    print(f"Kept (missing data):  {kept_count}")
    print(f"Already deleted:      {missing_count}")
    print(f"Errors:               {error_count}")
    print(f"{'='*70}")

    if dry_run:
        print("\nThis was a DRY RUN. No files were actually deleted.")
        print("Run with dry_run=False to perform actual deletion.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Clean up DENIED image files')
    parser.add_argument('--live', action='store_true',
                       help='Actually delete files (default is dry-run)')
    args = parser.parse_args()

    cleanup_denied_images(dry_run=not args.live)
