#!/usr/bin/env python3
"""
Test client for Content Moderation API.
Demonstrates how to call the Unix socket API from Python.
"""
import requests
import requests_unixsocket
import time
import base64
import json

# Register Unix socket support
requests_unixsocket.monkeypatch()


class ModeratorClient:
    """Client for content moderation API."""

    def __init__(self, socket_path="/tmp/moderator.sock"):
        self.socket_path = socket_path
        # Encode socket path for requests
        # /tmp/moderator.sock -> http+unix://%2Ftmp%2Fmoderator.sock/
        encoded_socket = socket_path.replace("/", "%2F")
        self.base_url = f"http+unix://{encoded_socket}"

    def health_check(self):
        """Check if API is healthy."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

    def check_text(self, text: str):
        """Check text content."""
        response = requests.post(
            f"{self.base_url}/check/text",
            json={"text": text}
        )
        return response.json()

    def check_image(self, image_bytes: bytes):
        """Check image content."""
        # Encode image to base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        response = requests.post(
            f"{self.base_url}/check/image",
            json={"image_base64": image_b64}
        )
        return response.json()

    def get_stats(self):
        """Get moderation statistics."""
        response = requests.get(f"{self.base_url}/stats")
        return response.json()


def test_text_moderation():
    """Test text content moderation."""
    client = ModeratorClient()

    print("="*70)
    print("TEXT MODERATION TEST")
    print("="*70)

    test_texts = [
        "I hate all Jews and they should be eliminated",  # Should match hate speech
        "The weather is nice today",  # Should be allowed
        "Hitler did nothing wrong",  # Should match hate speech
        "I love puppies",  # Should be allowed
    ]

    for text in test_texts:
        start = time.time()
        result = client.check_text(text)
        elapsed = (time.time() - start) * 1000

        print(f"\nText: \"{text[:50]}...\"")
        print(f"  Status: {result['status']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Match type: {result['match_type']}")
        print(f"  API time: {elapsed:.1f}ms")
        print(f"  Lookup time: {result['lookup_time_ms']:.1f}ms")

        if result['reason']:
            print(f"  Reason: {result['reason'][:100]}")


def test_image_moderation():
    """Test image content moderation."""
    client = ModeratorClient()

    print("\n" + "="*70)
    print("IMAGE MODERATION TEST")
    print("="*70)

    # Try to find a test image
    import config
    import sqlite3

    db = sqlite3.connect(config.DB_PATH)
    cursor = db.execute("""
        SELECT encoded_path, gemini_label
        FROM posts
        WHERE encoded_path IS NOT NULL
        AND gemini_label IS NOT NULL
        LIMIT 5
    """)

    for encoded_path, label in cursor:
        try:
            with open(encoded_path, 'rb') as f:
                image_bytes = f.read()

            start = time.time()
            result = client.check_image(image_bytes)
            elapsed = (time.time() - start) * 1000

            print(f"\nImage: {encoded_path}")
            print(f"  Expected: {label}")
            print(f"  Status: {result['status']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Match type: {result['match_type']}")
            print(f"  API time: {elapsed:.1f}ms")
            print(f"  Lookup time: {result['lookup_time_ms']:.1f}ms")

        except Exception as e:
            print(f"\nError testing {encoded_path}: {e}")

    db.close()


def benchmark_api():
    """Benchmark API performance."""
    client = ModeratorClient()

    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARK")
    print("="*70)

    test_text = "I hate all Jews"

    # Warmup
    for _ in range(5):
        client.check_text(test_text)

    # Measure
    times = []
    for _ in range(100):
        start = time.time()
        client.check_text(test_text)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    import numpy as np
    print(f"\n100 API calls:")
    print(f"  Mean: {np.mean(times):.2f}ms")
    print(f"  Median: {np.median(times):.2f}ms")
    print(f"  P95: {np.percentile(times, 95):.2f}ms")
    print(f"  P99: {np.percentile(times, 99):.2f}ms")
    print(f"  Min: {np.min(times):.2f}ms")
    print(f"  Max: {np.max(times):.2f}ms")


def main():
    """Run all tests."""
    try:
        client = ModeratorClient()

        # Health check
        print("Checking API health...")
        health = client.health_check()
        print(f"✓ API is healthy")
        print(f"  Text index: {health['text_vectors']:,} vectors")
        print(f"  Image index: {health['image_vectors']:,} vectors")
        print()

        # Run tests
        test_text_moderation()
        test_image_moderation()
        benchmark_api()

    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API")
        print("  Make sure the API is running:")
        print("  python moderator_api.py")


if __name__ == "__main__":
    main()
