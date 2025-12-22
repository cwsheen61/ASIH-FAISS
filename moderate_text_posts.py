#!/usr/bin/env python3
"""
Moderate text-only posts using Ollama.
Processes quarantined text posts and generates fingerprints.
"""
import sqlite3
import time
from ollama_text_moderator import OllamaTextModerator
from text_fingerprint import generate_text_fingerprint
import config

def moderate_text_posts(limit: int = None, batch_size: int = 50):
    """
    Moderate text-only posts with Ollama.

    Args:
        limit: Maximum number of posts to process (None = all)
        batch_size: Progress update frequency
    """
    print("="*70)
    print("TEXT-ONLY POST MODERATION WITH OLLAMA")
    print("="*70)

    moderator = OllamaTextModerator()

    # Get text-only posts that need moderation
    db = sqlite3.connect(config.DB_PATH)
    cursor = db.execute("""
        SELECT id, post_id, content_text, source
        FROM posts
        WHERE quarantined = 1
        AND post_type = 'text'
        AND content_text IS NOT NULL
        AND LENGTH(content_text) > 10
        AND gemini_label IS NULL
        ORDER BY id DESC
        {}
    """.format(f"LIMIT {limit}" if limit else ""))

    text_posts = cursor.fetchall()
    db.close()

    total = len(text_posts)
    print(f"\n✓ Found {total} text-only posts to moderate")

    if total == 0:
        print("Nothing to do!")
        return

    # Process posts
    stats = {
        'processed': 0,
        'denied': 0,
        'allowed': 0,
        'failed': 0
    }

    print("\nStarting moderation...\n")

    for idx, (db_id, post_id, text, source) in enumerate(text_posts, 1):
        # Progress
        if idx % batch_size == 0:
            print(f"Progress: {idx}/{total} ({idx/total*100:.1f}%) - "
                  f"Denied: {stats['denied']}, Allowed: {stats['allowed']}, Failed: {stats['failed']}")

        # Moderate with Ollama
        result = moderator.classify_text(text)

        if not result:
            stats['failed'] += 1
            continue

        label, reason = result

        # Generate text fingerprints
        text_hash, text_embedding = generate_text_fingerprint(text, "")

        # Update database
        db = sqlite3.connect(config.DB_PATH)
        db.execute("""
            UPDATE posts
            SET gemini_label = ?,
                gemini_reason = ?,
                text_hash = ?,
                text_embedding = ?,
                gemini_checked_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (label, reason, text_hash, text_embedding, db_id))
        db.commit()
        db.close()

        # Update stats
        stats['processed'] += 1
        if label == 'DENIED':
            stats['denied'] += 1
        else:
            stats['allowed'] += 1

    # Final summary
    print("\n" + "="*70)
    print("MODERATION COMPLETE")
    print("="*70)

    print(f"\nTotal processed: {stats['processed']}")
    print(f"  DENIED:  {stats['denied']} ({stats['denied']/stats['processed']*100 if stats['processed'] > 0 else 0:.1f}%)")
    print(f"  ALLOWED: {stats['allowed']} ({stats['allowed']/stats['processed']*100 if stats['processed'] > 0 else 0:.1f}%)")
    print(f"  Failed:  {stats['failed']}")

    print("\n✓ Text fingerprints generated for all moderated posts")
    print("✓ Ready for SITF/ASIH similarity matching")
    print("="*70)


if __name__ == "__main__":
    import sys

    # Check for limit argument
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
            print(f"Processing {limit} posts (test mode)")
        except:
            print(f"Usage: python {sys.argv[0]} [limit]")
            sys.exit(1)

    moderate_text_posts(limit=limit)
