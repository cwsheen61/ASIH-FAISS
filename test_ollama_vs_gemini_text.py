#!/usr/bin/env python3
"""
Compare Ollama (local) vs Gemini (API) for text-only moderation.
Test if local model is good enough to save API costs.
"""
import sqlite3
import time
import requests
import json
from gemini_labeler import GeminiLabeler
import config

class OllamaTextModerator:
    """Local text moderation using Ollama."""

    def __init__(self, model: str = "content-moderator"):
        self.model = model
        self.base_url = "http://localhost:11434"

    def classify_text(self, text: str):
        """
        Classify text as DENIED or ALLOWED.

        Returns:
            Tuple of (label, reason) or None if failed
        """
        prompt = f"""You are a content moderation classifier. Analyze this text and determine if it violates community guidelines.

CLASSIFY AS DENIED IF THE TEXT CONTAINS:
- Racist slurs or content targeting race/ethnicity
- Homophobic or transphobic slurs/content
- Antisemitic content or Jewish stereotypes
- Anti-Muslim or Islamophobic content
- Hate speech targeting any protected group
- Calls for violence or harm
- Sexual or explicit content

CLASSIFY AS ALLOWED IF THE TEXT:
- Is normal discussion without hate speech
- Contains political opinions without bigotry
- Is informative content

TEXT TO ANALYZE:
{text}

Respond ONLY with valid JSON in this exact format:
{{"classification": "DENIED", "reason": "specific reason"}}
or
{{"classification": "ALLOWED", "reason": "why it's acceptable"}}"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistency
                    }
                },
                timeout=30
            )

            if response.status_code != 200:
                return None

            result = response.json()
            text_response = result.get('response', '')

            # Try to extract JSON from response
            # Ollama might wrap it in markdown or add extra text
            start = text_response.find('{')
            end = text_response.rfind('}') + 1

            if start != -1 and end > start:
                json_str = text_response[start:end]
                data = json.loads(json_str)

                classification = data.get('classification', '').upper()
                reason = data.get('reason', '')

                if classification in ['DENIED', 'ALLOWED']:
                    return (classification, reason)

            return None

        except Exception as e:
            print(f"Ollama error: {e}")
            return None

def test_text_moderation():
    print("="*70)
    print("OLLAMA vs GEMINI - Text-Only Moderation Comparison")
    print("="*70)

    # Get sample of text-only posts
    db = sqlite3.connect(config.DB_PATH)
    cursor = db.execute("""
        SELECT id, post_id, content_text
        FROM posts
        WHERE quarantined = 1
        AND post_type = 'text'
        AND content_text IS NOT NULL
        AND LENGTH(content_text) > 20
        ORDER BY RANDOM()
        LIMIT 25
    """)
    text_posts = cursor.fetchall()
    db.close()

    print(f"\n✓ Sampled {len(text_posts)} text-only posts")

    # Initialize moderators
    print("\nInitializing moderators...")

    # Check if Ollama is available
    ollama = OllamaTextModerator(model="content-moderator")
    try:
        test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if test_response.status_code == 200:
            models = test_response.json().get('models', [])
            available_models = [m['name'] for m in models]
            print(f"✓ Ollama available with models: {available_models[:3]}...")
            if 'content-moderator:latest' in available_models or 'content-moderator' in str(available_models):
                print("  Using content-moderator (custom trained model)")
            else:
                print(f"  WARNING: content-moderator not found, using first available")
                if available_models:
                    ollama.model = available_models[0]
        else:
            print("✗ Ollama not responding")
            return
    except Exception as e:
        print(f"✗ Ollama not available: {e}")
        print("  Start with: ollama serve")
        return

    gemini = GeminiLabeler()
    print("✓ Gemini API ready")

    # Test both on same texts
    print("\n" + "="*70)
    print("MODERATION COMPARISON")
    print("="*70)

    results = {
        'ollama_only': [],
        'gemini_only': [],
        'both_denied': [],
        'both_allowed': [],
        'disagree': []
    }

    ollama_time = 0
    gemini_time = 0

    for db_id, post_id, text in text_posts:
        clean_text = text[:200].replace('\n', ' ')
        print(f"\n--- Post {post_id} ---")
        print(f"Text: {clean_text}...")

        # Ollama classification
        print("  Ollama: ", end="", flush=True)
        start = time.time()
        ollama_result = ollama.classify_text(text)
        ollama_time += time.time() - start

        if ollama_result:
            ollama_label, ollama_reason = ollama_result
            print(f"{ollama_label} ({time.time() - start:.2f}s)")
        else:
            print("FAILED")
            ollama_label = None

        # Gemini classification
        print("  Gemini: ", end="", flush=True)
        start = time.time()
        gemini_result = gemini.classify_content(text, encoded_image_path=None)
        gemini_time += time.time() - start

        if gemini_result:
            gemini_label, gemini_reason = gemini_result
            print(f"{gemini_label} ({time.time() - start:.2f}s)")
        else:
            print("FAILED")
            gemini_label = None

        # Categorize result
        if ollama_label and gemini_label:
            if ollama_label == gemini_label:
                if ollama_label == 'DENIED':
                    results['both_denied'].append(post_id)
                    print("  → ✓ AGREEMENT: Both flagged as hate")
                else:
                    results['both_allowed'].append(post_id)
                    print("  → ✓ AGREEMENT: Both allowed")
            else:
                results['disagree'].append({
                    'post_id': post_id,
                    'ollama': ollama_label,
                    'gemini': gemini_label,
                    'text': clean_text
                })
                print(f"  → ✗ DISAGREEMENT: Ollama={ollama_label}, Gemini={gemini_label}")

        time.sleep(1.0)  # Rate limiting

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    total = len(text_posts)
    agreed = len(results['both_denied']) + len(results['both_allowed'])
    disagreed = len(results['disagree'])

    print(f"\nTotal texts tested: {total}")
    print(f"\nAgreement:")
    print(f"  Both DENIED:  {len(results['both_denied'])} ({len(results['both_denied'])/total*100:.1f}%)")
    print(f"  Both ALLOWED: {len(results['both_allowed'])} ({len(results['both_allowed'])/total*100:.1f}%)")
    print(f"  Total agreement: {agreed}/{total} ({agreed/total*100:.1f}%)")

    print(f"\nDisagreement:")
    print(f"  Conflicting labels: {disagreed}/{total} ({disagreed/total*100:.1f}%)")

    if results['disagree']:
        print("\n  Examples of disagreement:")
        for item in results['disagree'][:3]:
            print(f"    Post {item['post_id']}:")
            print(f"      Ollama: {item['ollama']}")
            print(f"      Gemini: {item['gemini']}")
            print(f"      Text: {item['text'][:80]}...")

    print(f"\nPerformance:")
    print(f"  Ollama avg time: {ollama_time/total:.2f}s per text")
    print(f"  Gemini avg time: {gemini_time/total:.2f}s per text")
    print(f"  Speedup: {gemini_time/ollama_time:.1f}x faster" if ollama_time > 0 else "")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    accuracy = agreed / total * 100 if total > 0 else 0

    if accuracy >= 90:
        print(f"\n✓ HIGH AGREEMENT ({accuracy:.1f}%)")
        print("  Ollama is accurate enough for text-only moderation")
        print("  Recommendation: Use Ollama for text, save Gemini for images")
    elif accuracy >= 75:
        print(f"\n⚠ MODERATE AGREEMENT ({accuracy:.1f}%)")
        print("  Ollama catches most hate but misses some")
        print("  Recommendation: Use Ollama as first pass, Gemini for uncertain cases")
    else:
        print(f"\n✗ LOW AGREEMENT ({accuracy:.1f}%)")
        print("  Ollama not reliable enough")
        print("  Recommendation: Use Gemini for all moderation")

    print("="*70)

if __name__ == "__main__":
    test_text_moderation()
