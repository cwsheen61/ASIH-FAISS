#!/usr/bin/env python3
"""
Check if quarantined duplicates are reposts of labeled content.
"""
import sqlite3
from collections import defaultdict
import config

def main():
    print("="*70)
    print("DUPLICATE REPOST ANALYSIS")
    print("="*70)

    db = sqlite3.connect(config.DB_PATH)

    # Get ALL posts (quarantined and not) grouped by pHash
    cursor = db.execute("""
        SELECT post_id, image_phash, quarantined, gemini_label
        FROM posts
        WHERE image_phash IS NOT NULL
        ORDER BY id ASC
    """)
    all_posts = cursor.fetchall()

    print(f"\n✓ Loaded {len(all_posts)} posts with pHash")

    # Group by pHash
    phash_groups = defaultdict(list)
    for post_id, phash, quarantined, label in all_posts:
        phash_groups[phash].append({
            'post_id': post_id,
            'quarantined': quarantined,
            'label': label
        })

    # Find groups with duplicates
    duplicate_groups = {k: v for k, v in phash_groups.items() if len(v) > 1}

    print(f"✓ Found {len(duplicate_groups)} pHash groups with duplicates")

    # Analyze each duplicate group
    denied_repost_groups = []
    allowed_repost_groups = []
    unlabeled_repost_groups = []

    total_reposts = 0

    for phash, posts in duplicate_groups.items():
        # Count original vs reposts
        non_quarantined = [p for p in posts if not p['quarantined']]
        quarantined = [p for p in posts if p['quarantined']]

        if not non_quarantined:
            # All copies were quarantined (shouldn't happen but check)
            continue

        original = non_quarantined[0]  # First non-quarantined is the original
        num_reposts = len(posts) - 1  # Everyone else is a repost
        total_reposts += num_reposts

        label = original['label']

        if label == 'DENIED':
            denied_repost_groups.append({
                'original': original['post_id'],
                'num_reposts': num_reposts,
                'repost_ids': [p['post_id'] for p in posts[1:]]
            })
        elif label == 'ALLOWED':
            allowed_repost_groups.append({
                'original': original['post_id'],
                'num_reposts': num_reposts
            })
        else:
            unlabeled_repost_groups.append({
                'original': original['post_id'],
                'num_reposts': num_reposts
            })

    db.close()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    total_denied_reposts = sum(g['num_reposts'] for g in denied_repost_groups)
    total_allowed_reposts = sum(g['num_reposts'] for g in allowed_repost_groups)
    total_unlabeled_reposts = sum(g['num_reposts'] for g in unlabeled_repost_groups)

    print(f"\nTotal repost instances: {total_reposts}")
    print(f"\nBreakdown by original label:")
    print(f"  DENIED originals reposted: {len(denied_repost_groups)} images, {total_denied_reposts} total reposts")
    print(f"  ALLOWED originals reposted: {len(allowed_repost_groups)} images, {total_allowed_reposts} total reposts")
    print(f"  Unlabeled originals reposted: {len(unlabeled_repost_groups)} images, {total_unlabeled_reposts} total reposts")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    if total_denied_reposts > 0:
        print(f"\n✓ YES - Hateful content IS being reposted!")
        print(f"  {len(denied_repost_groups)} DENIED images were reposted {total_denied_reposts} times")
        print(f"\nThis validates your hypothesis:")
        print(f"  - Hateful memes get copied across 4chan threads")
        print(f"  - We're catching exact duplicates with pHash (quarantining {total_denied_reposts} hate reposts)")
        print(f"  - CLIP similarity could catch NEAR-duplicates (cropped/edited versions)")
    else:
        print(f"\n✗ No DENIED reposts detected yet")
        print(f"  This might mean:")
        print(f"  1. We haven't labeled enough images yet ({len(unlabeled_repost_groups)} repost groups unlabeled)")
        print(f"  2. Or hateful content isn't being reposted in our dataset")

    # Show examples
    if denied_repost_groups:
        print(f"\nExamples of DENIED content being reposted:")
        for group in denied_repost_groups[:5]:
            print(f"  Original {group['original']} reposted {group['num_reposts']} time(s)")
            print(f"    Reposts: {group['repost_ids'][:3]}")

    print("="*70)

if __name__ == "__main__":
    main()
