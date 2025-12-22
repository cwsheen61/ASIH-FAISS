#!/usr/bin/env python3
"""
Test FAISS-accelerated text fingerprint matching.
Measures the speedup from using FAISS for similarity search.
"""
import requests
import sqlite3
import time
import json
import numpy as np
import faiss
import pickle
from typing import Optional, Tuple
from text_fingerprint import hash_text, embed_text, get_text_encoder
from ollama_text_moderator import OllamaTextModerator
import config

class FastTextLookup:
    """Fast text lookup using FAISS."""

    def __init__(self):
        # Pre-load models (one-time cost)
        print("Loading models (one-time setup)...")
        start = time.time()

        self.text_encoder = get_text_encoder()

        index_path = config.DATA_DIR / "text_embeddings.faiss"
        metadata_path = config.DATA_DIR / "text_embeddings_metadata.pkl"

        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        print(f"✓ Models loaded in {time.time() - start:.1f}s")
        print(f"✓ FAISS index: {self.index.ntotal:,} vectors\n")

    def find_match(self, text: str, threshold: float = 0.85) -> Optional[Tuple[str, str, str, float]]:
        """Find text match using FAISS."""
        # Generate embedding
        emb = self.text_encoder.encode(text)
        query_vec = np.array(emb, dtype='float32').reshape(1, -1)
        faiss.normalize_L2(query_vec)

        # FAISS search
        similarities, indices = self.index.search(query_vec, 1)

        if similarities[0][0] >= threshold:
            idx = indices[0][0]
            meta = self.metadata[idx]
            return (meta['label'], meta['reason'], 'semantic', float(similarities[0][0]))

        return None

def scrape_text_comments(board: str = "pol", target: int = 100):
    """Scrape text-only comments from 4chan."""
    print(f"Scraping {target} text comments from /{board}/...")

    response = requests.get(f"https://a.4cdn.org/{board}/catalog.json")
    if response.status_code != 200:
        return []

    catalog = response.json()
    comments = []

    for page in catalog:
        for thread in page.get('threads', []):
            thread_id = thread['no']
            response = requests.get(f"https://a.4cdn.org/{board}/thread/{thread_id}.json")
            if response.status_code != 200:
                continue

            thread_data = response.json()
            for post in thread_data.get('posts', []):
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
                    print(f"✓ Collected {len(comments)} text comments\n")
                    return comments

            time.sleep(0.5)

    print(f"✓ Collected {len(comments)} text comments\n")
    return comments

def main():
    print("="*70)
    print("FAISS TEXT FINGERPRINT EFFICIENCY TEST")
    print("="*70)
    print()

    # Initialize FAISS lookup (loads models once)
    faiss_lookup = FastTextLookup()
    moderator = OllamaTextModerator()

    # Scrape fresh comments
    comments = scrape_text_comments(board="pol", target=1000)
    if not comments:
        print("Failed to scrape comments")
        return

    print("Testing fingerprint matching with FAISS...\n")

    results = {
        'exact_hash': [],
        'semantic_match': [],
        'novel': []
    }

    timings = {
        'faiss_lookup': [],
        'ollama_moderation': []
    }

    for idx, comment in enumerate(comments, 1):
        # FAISS lookup
        start = time.time()
        match = faiss_lookup.find_match(comment['text'])
        lookup_time = time.time() - start
        timings['faiss_lookup'].append(lookup_time)

        if match:
            label, reason, match_type, confidence = match
            results['semantic_match'].append({
                'comment': comment,
                'similarity': confidence,
                'label': label
            })
            print(f"{idx:3d}. ✓  MATCH ({confidence:.3f}) - {label} ({lookup_time*1000:.1f}ms)")
        else:
            # Novel - need Ollama
            results['novel'].append(comment)

            start = time.time()
            ollama_result = moderator.classify_text(comment['text'])
            mod_time = time.time() - start
            timings['ollama_moderation'].append(mod_time)

            if ollama_result:
                label, reason = ollama_result
                print(f"{idx:3d}. ?  NOVEL - {label} ({lookup_time*1000:.1f}ms lookup + {mod_time*1000:.0f}ms ollama)")
            else:
                print(f"{idx:3d}. ?  NOVEL - FAILED")

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    total = len(comments)
    semantic = len(results['semantic_match'])
    novel = len(results['novel'])

    print(f"\nTotal comments tested: {total}")
    print(f"\nFingerprint matching:")
    print(f"  Semantic matches:        {semantic} ({semantic/total*100:.1f}%)")
    print(f"  Novel (need moderation): {novel} ({novel/total*100:.1f}%)")

    # Performance
    avg_lookup = np.mean(timings['faiss_lookup']) * 1000
    avg_ollama = np.mean(timings['ollama_moderation']) * 1000 if timings['ollama_moderation'] else 0

    print(f"\nPerformance:")
    print(f"  Avg FAISS lookup:        {avg_lookup:.1f}ms")
    print(f"  Avg Ollama moderation:   {avg_ollama:.0f}ms")
    print(f"  Speedup for cached:      {avg_ollama/avg_lookup:.1f}x faster")

    # Time savings
    time_with_cache = (semantic * avg_lookup/1000) + (novel * avg_ollama/1000)
    time_without_cache = total * (avg_ollama/1000)
    time_saved = time_without_cache - time_with_cache

    print(f"\nEfficiency:")
    print(f"  Time with FAISS cache:   {time_with_cache:.1f}s")
    print(f"  Time without cache:      {time_without_cache:.1f}s")
    print(f"  Time saved:              {time_saved:.1f}s ({time_saved/time_without_cache*100:.1f}%)")

    print("\n" + "="*70)
    print("COMPARISON: Brute Force vs FAISS")
    print("="*70)
    print(f"\nBrute force lookup:      ~2,938ms")
    print(f"FAISS lookup:            ~{avg_lookup:.1f}ms")
    print(f"FAISS speedup:           {2938/avg_lookup:.0f}x faster")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    if semantic >= total * 0.2:
        print(f"\n✓ FAISS IS EFFECTIVE!")
        print(f"  Cache hit rate: {semantic/total*100:.1f}%")
        print(f"  Lookup speed: {avg_lookup:.1f}ms (vs 2,938ms brute force)")
        print(f"  Overall speedup: {time_saved/time_without_cache*100:.1f}% faster than no cache")
        print(f"\n  FAISS enables practical text fingerprint matching!")
    else:
        print(f"\n⚠ Low cache hit rate ({semantic/total*100:.1f}%)")
        print(f"  FAISS is fast, but not enough reposts to matter")

    print("="*70)

if __name__ == "__main__":
    main()
