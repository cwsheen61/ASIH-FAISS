#!/usr/bin/env python3
"""
Check if quarantined duplicate images are reposts of DENIED content.
"""
import sqlite3
import config

def main():
    print("="*70)
    print("QUARANTINED DUPLICATE ANALYSIS")
    print("="*70)

    db = sqlite3.connect(config.DB_PATH)

    # Get all quarantined duplicates
    cursor = db.execute("""
        SELECT post_id, image_phash
        FROM posts
        WHERE quarantined = 1
        AND has_image = 1
    """)
    quarantined = cursor.fetchall()

    print(f"\n✓ Found {len(quarantined)} quarantined duplicate images")

    # For each, find the original
    denied_reposts = 0
    allowed_reposts = 0
    unlabeled_reposts = 0

    examples_denied = []
    examples_allowed = []

    for post_id, phash in quarantined:
        if not phash:
            continue

        # Find the original post with this pHash
        cursor = db.execute("""
            SELECT post_id, gemini_label, gemini_reason
            FROM posts
            WHERE image_phash = ?
            AND quarantined = 0
            LIMIT 1
        """, (phash,))
        original = cursor.fetchone()

        if original:
            orig_post_id, label, label_reason = original

            if label == 'DENIED':
                denied_reposts += 1
                if len(examples_denied) < 5:
                    examples_denied.append({
                        'original': orig_post_id,
                        'repost': post_id,
                        'reason': label_reason[:80] if label_reason else 'N/A'
                    })
            elif label == 'ALLOWED':
                allowed_reposts += 1
                if len(examples_allowed) < 3:
                    examples_allowed.append({
                        'original': orig_post_id,
                        'repost': post_id,
                        'reason': label_reason[:80] if label_reason else 'N/A'
                    })
            else:
                unlabeled_reposts += 1

    db.close()

    print(f"\nBreakdown of quarantined duplicates:")
    print(f"  Reposts of DENIED content: {denied_reposts}")
    print(f"  Reposts of ALLOWED content: {allowed_reposts}")
    print(f"  Reposts of unlabeled content: {unlabeled_reposts}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    if denied_reposts > 0:
        print(f"\n✓ YES - We ARE seeing hateful content reposted!")
        print(f"  {denied_reposts} reposts of DENIED images were caught by pHash deduplication")
        print(f"\nThis validates your point: hateful memes get copied across threads.")
        print(f"Our pHash deduplication is already catching these reposts.")
    else:
        print(f"\n✗ No DENIED reposts detected yet")

    # Show examples
    if examples_denied:
        print(f"\nExamples of DENIED reposts:")
        for ex in examples_denied:
            print(f"  Original Post {ex['original']} reposted as {ex['repost']}")
            print(f"    Reason: {ex['reason']}")

    if examples_allowed:
        print(f"\nExamples of ALLOWED reposts:")
        for ex in examples_allowed:
            print(f"  Original Post {ex['original']} reposted as {ex['repost']}")

    print("\n" + "="*70)
    print("IMPLICATION FOR CLIP SIMILARITY LOOKUP")
    print("="*70)
    print(f"\npHash catches EXACT duplicates: {denied_reposts + allowed_reposts} images")
    print(f"CLIP similarity could catch NEAR-duplicates (cropped/edited versions)")
    print(f"\nFor the similarity lookup to be useful, we need to detect:")
    print(f"  - Slightly cropped versions")
    print(f"  - Re-compressed/resized versions")
    print(f"  - Images with text overlays added/removed")
    print("\nThese would have high CLIP similarity (>0.95) but different pHash")
    print("="*70)

if __name__ == "__main__":
    main()
