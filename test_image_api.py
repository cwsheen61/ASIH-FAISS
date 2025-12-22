#!/usr/bin/env python3
"""
Test image moderation API with real images from database.
"""
import requests_unixsocket
import sqlite3
import base64
import time
import json
import config

requests_unixsocket.monkeypatch()

class ModeratorClient:
    def __init__(self, socket_path="/tmp/moderator.sock"):
        encoded_socket = socket_path.replace("/", "%2F")
        self.base_url = f"http+unix://{encoded_socket}"

    def check_image(self, image_bytes: bytes):
        """Check image content."""
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        response = requests_unixsocket.post(
            f"{self.base_url}/check/image",
            json={"image_base64": image_b64}
        )
        return response.json()

    def check_text(self, text: str):
        """Check text content."""
        response = requests_unixsocket.post(
            f"{self.base_url}/check/text",
            json={"text": text}
        )
        return response.json()

def test_image_similarity():
    """Test image similarity with images from database."""
    client = ModeratorClient()

    print("="*70)
    print("IMAGE SIMILARITY TEST")
    print("="*70)

    # Get some labeled images from database
    db = sqlite3.connect(config.DB_PATH)
    cursor = db.execute("""
        SELECT encoded_path, gemini_label, gemini_reason
        FROM posts
        WHERE encoded_path IS NOT NULL
        AND clip_embedding IS NOT NULL
        AND gemini_label IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 10
    """)

    results = {
        'cache_hit': 0,
        'cache_miss': 0,
        'total': 0
    }

    for encoded_path, expected_label, reason in cursor:
        try:
            # Read the base64-encoded image and decode it
            with open(encoded_path, 'rb') as f:
                image_b64 = f.read()

            # Decode from base64 to get raw image bytes
            image_bytes = base64.b64decode(image_b64)

            # Test API
            start = time.time()
            result = client.check_image(image_bytes)
            elapsed = (time.time() - start) * 1000

            results['total'] += 1

            if result['match_type'] == 'semantic':
                results['cache_hit'] += 1
                match_status = "✓ HIT"
            else:
                results['cache_miss'] += 1
                match_status = "✗ MISS"

            print(f"\n{match_status} - {encoded_path.split('/')[-1][:40]}")
            print(f"  Expected: {expected_label}")
            print(f"  Got: {result['status']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  API time: {elapsed:.1f}ms")

            if result['match_type']:
                print(f"  Match type: {result['match_type']}")

        except Exception as e:
            print(f"\n✗ Error testing {encoded_path}: {e}")

    db.close()

    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nTotal tested: {results['total']}")
    print(f"Cache hits: {results['cache_hit']} ({results['cache_hit']/results['total']*100:.1f}%)")
    print(f"Cache misses: {results['cache_miss']} ({results['cache_miss']/results['total']*100:.1f}%)")

    if results['cache_hit'] == results['total']:
        print("\n✓ PERFECT! All images matched in FAISS index")
    elif results['cache_hit'] > 0:
        print(f"\n⚠ Some images matched ({results['cache_hit']}/{results['total']})")
    else:
        print("\n✗ No images matched - threshold may be too high")

def test_text_similarity():
    """Test text similarity with text from database."""
    client = ModeratorClient()

    print("\n" + "="*70)
    print("TEXT SIMILARITY TEST")
    print("="*70)

    # Get some labeled text from database
    db = sqlite3.connect(config.DB_PATH)
    cursor = db.execute("""
        SELECT content_text, gemini_label, gemini_reason
        FROM posts
        WHERE content_text IS NOT NULL
        AND text_embedding IS NOT NULL
        AND gemini_label IS NOT NULL
        AND content_text != ''
        ORDER BY RANDOM()
        LIMIT 10
    """)

    results = {
        'cache_hit': 0,
        'cache_miss': 0,
        'total': 0
    }

    for content, expected_label, reason in cursor:
        try:
            # Test API
            start = time.time()
            result = client.check_text(content)
            elapsed = (time.time() - start) * 1000

            results['total'] += 1

            if result['match_type'] == 'semantic':
                results['cache_hit'] += 1
                match_status = "✓ HIT"
            else:
                results['cache_miss'] += 1
                match_status = "✗ MISS"

            print(f"\n{match_status} - \"{content[:50]}...\"")
            print(f"  Expected: {expected_label}")
            print(f"  Got: {result['status']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  API time: {elapsed:.1f}ms")

            if result['match_type']:
                print(f"  Match type: {result['match_type']}")

        except Exception as e:
            print(f"\n✗ Error testing text: {e}")

    db.close()

    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nTotal tested: {results['total']}")
    print(f"Cache hits: {results['cache_hit']} ({results['cache_hit']/results['total']*100:.1f}%)")
    print(f"Cache misses: {results['cache_miss']} ({results['cache_miss']/results['total']*100:.1f}%)")

    if results['cache_hit'] == results['total']:
        print("\n✓ PERFECT! All text matched in FAISS index")
    elif results['cache_hit'] > 0:
        print(f"\n⚠ Some text matched ({results['cache_hit']}/{results['total']})")
    else:
        print("\n✗ No text matched - threshold may be too high")

def main():
    test_image_similarity()
    test_text_similarity()

if __name__ == "__main__":
    main()
