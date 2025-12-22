#!/usr/bin/env python3
"""
Test text fingerprint matching efficiency.
Scrapes 100 new text-only comments and measures how many are caught by
semantic matching vs. needing fresh moderation.
"""
import requests
import sqlite3
import time
import json
import numpy as np
from typing import Optional, Tuple
from text_fingerprint import hash_text, embed_text, text_similarity
from ollama_text_moderator import OllamaTextModerator
import config

def scrape_text_comments(board: str = "pol", target: int = 100):
    """Scrape text-only comments from 4chan."""
    print(f"Scraping {target} text comments from /{board}/...")

    # Get catalog
    response = requests.get(f"https://a.4cdn.org/{board}/catalog.json")
    if response.status_code != 200:
        print("Failed to fetch catalog")
        return []

    catalog = response.json()

    comments = []

    for page in catalog:
        for thread in page.get('threads', []):
            thread_id = thread['no']

            # Fetch thread
            response = requests.get(f"https://a.4cdn.org/{board}/thread/{thread_id}.json")
            if response.status_code != 200:
                continue

            thread_data = response.json()

            for post in thread_data.get('posts', []):
                # Only text-only posts (no image)
                if 'tim' in post:
                    continue

                text = post.get('com', '')
                if not text or len(text) < 20:
                    continue

                comments.append({
                    'post_id': post['no'],
                    'text': text,
                    'thread_id': thread_id
                })

                if len(comments) >= target:
                    print(f"✓ Collected {len(comments)} text comments")
                    return comments

            time.sleep(0.5)  # Rate limiting

    print(f"✓ Collected {len(comments)} text comments")
    return comments

def find_text_match(text: str, db_conn) -> Optional[Tuple[str, str, str, float]]:
    """
    Find matching text in fingerprint database.

    Returns:
        Tuple of (label, reason, match_type, confidence) or None
    """
    # Generate fingerprints
    text_hash = hash_text(text)
    text_emb = embed_text(text)

    if not text_hash and not text_emb:
        return None

    # Step 1: Exact hash match (instant)
    if text_hash:
        cursor = db_conn.execute("""
            SELECT gemini_label, gemini_reason
            FROM posts
            WHERE text_hash = ?
            AND gemini_label IS NOT NULL
            LIMIT 1
        """, (text_hash,))

        row = cursor.fetchone()
        if row:
            return (row[0], row[1], 'exact_hash', 1.0)

    # Step 2: Semantic similarity (embedding)
    if text_emb:
        cursor = db_conn.execute("""
            SELECT text_embedding, gemini_label, gemini_reason
            FROM posts
            WHERE text_embedding IS NOT NULL
            AND gemini_label IS NOT NULL
        """)

        new_vec = np.array(json.loads(text_emb))
        best_match = None
        best_similarity = 0.85  # Threshold

        for existing_emb, label, reason in cursor:
            if not existing_emb:
                continue

            existing_vec = np.array(json.loads(existing_emb))
            similarity = text_similarity(text_emb, existing_emb)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (label, reason, 'semantic', similarity)

        if best_match:
            return best_match

    return None

def main():
    print("="*70)
    print("TEXT FINGERPRINT EFFICIENCY TEST")
    print("Testing SITF/ASIH semantic matching on fresh 4chan comments")
    print("="*70)

    # Scrape 100 new text comments
    comments = scrape_text_comments(board="pol", target=100)

    if not comments:
        print("Failed to scrape comments")
        return

    print(f"\n✓ Scraped {len(comments)} fresh text comments")
    print("\nTesting fingerprint matching...\n")

    # Connect to database
    db = sqlite3.connect(config.DB_PATH)

    # Test fingerprint matching
    results = {
        'exact_hash': [],
        'semantic_match': [],
        'novel': []
    }

    timings = {
        'fingerprint_lookup': [],
        'ollama_moderation': []
    }

    moderator = OllamaTextModerator()

    for idx, comment in enumerate(comments, 1):
        text = comment['text'][:200]  # Truncate for display

        # Test fingerprint lookup
        start = time.time()
        match = find_text_match(comment['text'], db)
        lookup_time = time.time() - start
        timings['fingerprint_lookup'].append(lookup_time)

        if match:
            label, reason, match_type, confidence = match

            if match_type == 'exact_hash':
                results['exact_hash'].append(comment)
                print(f"{idx:3d}. ✓✓ EXACT MATCH (hash) - {label} ({lookup_time*1000:.1f}ms)")
            else:
                results['semantic_match'].append({
                    'comment': comment,
                    'similarity': confidence,
                    'label': label
                })
                print(f"{idx:3d}. ✓  SEMANTIC MATCH ({confidence:.3f}) - {label} ({lookup_time*1000:.1f}ms)")
        else:
            # Novel - need to moderate
            results['novel'].append(comment)

            start = time.time()
            ollama_result = moderator.classify_text(comment['text'])
            mod_time = time.time() - start
            timings['ollama_moderation'].append(mod_time)

            if ollama_result:
                label, reason = ollama_result
                print(f"{idx:3d}. ?  NOVEL - {label} (lookup: {lookup_time*1000:.1f}ms, ollama: {mod_time*1000:.0f}ms)")
            else:
                print(f"{idx:3d}. ?  NOVEL - FAILED (lookup: {lookup_time*1000:.1f}ms)")

    db.close()

    # Calculate statistics
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    total = len(comments)
    exact = len(results['exact_hash'])
    semantic = len(results['semantic_match'])
    novel = len(results['novel'])
    cached = exact + semantic

    print(f"\nTotal comments tested: {total}")
    print(f"\nFingerprint matching:")
    print(f"  Exact matches (hash):    {exact} ({exact/total*100:.1f}%)")
    print(f"  Semantic matches:        {semantic} ({semantic/total*100:.1f}%)")
    print(f"  Total cached hits:       {cached} ({cached/total*100:.1f}%)")
    print(f"  Novel (need moderation): {novel} ({novel/total*100:.1f}%)")

    # Performance metrics
    avg_lookup = np.mean(timings['fingerprint_lookup']) * 1000
    avg_ollama = np.mean(timings['ollama_moderation']) * 1000 if timings['ollama_moderation'] else 0

    print(f"\nPerformance:")
    print(f"  Avg fingerprint lookup:  {avg_lookup:.1f}ms")
    print(f"  Avg Ollama moderation:   {avg_ollama:.0f}ms")
    print(f"  Speedup for cached:      {avg_ollama/avg_lookup:.1f}x faster")

    # Cost savings
    print(f"\nEfficiency gains:")
    print(f"  {cached} posts moderated instantly via fingerprints")
    print(f"  {novel} posts needed fresh moderation")
    print(f"  Computational savings:   {cached/total*100:.1f}% of posts skip AI inference")

    # Time savings
    time_with_cache = (cached * avg_lookup/1000) + (novel * avg_ollama/1000)
    time_without_cache = total * (avg_ollama/1000)
    time_saved = time_without_cache - time_with_cache

    print(f"  Time with fingerprints:  {time_with_cache:.1f}s")
    print(f"  Time without:            {time_without_cache:.1f}s")
    print(f"  Time saved:              {time_saved:.1f}s ({time_saved/time_without_cache*100:.1f}%)")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    if cached >= total * 0.5:
        print(f"\n✓ HIGHLY EFFECTIVE - {cached/total*100:.1f}% cache hit rate")
        print(f"  Text reposts are common on 4chan")
        print(f"  Fingerprint system provides major efficiency gains")
    elif cached >= total * 0.2:
        print(f"\n✓ MODERATELY EFFECTIVE - {cached/total*100:.1f}% cache hit rate")
        print(f"  Significant portion of content is reposted")
        print(f"  Fingerprints save meaningful computation")
    else:
        print(f"\n⚠ LIMITED EFFECTIVENESS - {cached/total*100:.1f}% cache hit rate")
        print(f"  Most content is novel")
        print(f"  Fingerprints help but fresh moderation still needed")

    print("="*70)

if __name__ == "__main__":
    main()
