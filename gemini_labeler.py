"""Gemini API integration for content labeling."""
import base64
import json
import time
from pathlib import Path
from typing import Optional, Tuple
from io import BytesIO
from PIL import Image
import requests
import config
from database import Database


class GeminiLabeler:
    """Uses Gemini API to label content as ALLOWED or DENIED."""

    def __init__(self, api_key: str = config.GEMINI_API_KEY):
        self.api_key = api_key
        self.db = Database()
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

    def decode_image(self, encoded_path: str) -> Optional[bytes]:
        """Decode base64 encoded image (already resized during scraping)."""
        try:
            with open(encoded_path, 'rb') as f:
                encoded_data = f.read()
            return base64.b64decode(encoded_data)
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None

    def get_image_mime_type(self, encoded_path: str) -> str:
        """Determine MIME type from file extension."""
        path = Path(encoded_path)
        ext = path.stem.split('.')[-1].lower()
        mime_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }
        return mime_types.get(ext, 'image/jpeg')

    def classify_content(self, text: Optional[str], encoded_image_path: Optional[str]) -> Optional[Tuple[str, str]]:
        """Classify content using Gemini API. Returns (label, reason) tuple."""
        if not self.api_key:
            print("Error: GEMINI_API_KEY not set")
            return None

        # Build request parts
        parts = [{"text": config.CLASSIFICATION_PROMPT}]

        if text:
            parts.append({"text": f"\n\nText content: {text}"})

        if encoded_image_path:
            image_data = self.decode_image(encoded_image_path)
            if image_data:
                mime_type = self.get_image_mime_type(encoded_image_path)
                parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(image_data).decode('utf-8')
                    }
                })

        payload = {
            "contents": [{
                "parts": parts
            }]
        }

        try:
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            # Extract the classification
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]

                # Check if content was blocked or filtered
                if 'content' not in candidate:
                    # Content was blocked - treat as DENIED
                    finish_reason = candidate.get('finishReason', 'UNKNOWN')
                    return ('DENIED', f'Content blocked by safety filter ({finish_reason})')

                text_response = candidate['content']['parts'][0]['text'].strip()

                # Try to parse JSON response
                try:
                    # Remove markdown code blocks if present
                    if text_response.startswith('```'):
                        lines = text_response.split('\n')
                        text_response = '\n'.join(lines[1:-1])

                    parsed = json.loads(text_response)
                    classification = parsed.get('classification', '').upper()
                    reason = parsed.get('reason', 'No reason provided')

                    if classification in ['ALLOWED', 'DENIED']:
                        return (classification, reason)
                    else:
                        print(f"Invalid classification: {classification}")
                        return None

                except json.JSONDecodeError:
                    # Fallback: try to extract from plain text
                    if 'DENIED' in text_response.upper():
                        return ('DENIED', text_response)
                    elif 'ALLOWED' in text_response.upper():
                        return ('ALLOWED', text_response)
                    else:
                        print(f"Could not parse response: {text_response}")
                        return None

        except requests.RequestException as e:
            print(f"Error calling Gemini API: {e}")
            return None

    def label_batch(self, batch_size: int = 10, delay: float = 1.0):
        """Label a batch of unlabeled posts."""
        unlabeled = self.db.get_unlabeled_posts(batch_size)

        if not unlabeled:
            print("No unlabeled posts found")
            return

        print(f"Labeling {len(unlabeled)} posts...")

        for post_id, text, encoded_path, has_image, phash in unlabeled:
            image_path = encoded_path if has_image else None

            print(f"  Classifying post {post_id}...", end=" ")

            # Check if we've already labeled an identical image
            if phash:
                existing_label = self.db.get_label_by_phash(phash)
                if existing_label:
                    label, reason = existing_label
                    self.db.update_gemini_label(post_id, label, reason)
                    print(f"{label} - {reason} [DUPLICATE]")
                    continue

            # No duplicate found, call API
            result = self.classify_content(text, image_path)

            if result:
                label, reason = result
                self.db.update_gemini_label(post_id, label, reason)
                print(f"{label} - {reason}")
            else:
                print("FAILED")

            time.sleep(delay)  # Rate limiting

        # Print stats
        stats = self.db.get_label_counts()
        print(f"\nLabeling stats: {stats}")
        total_labeled = sum(stats.values())
        print(f"Total labeled: {total_labeled}/{config.TRAINING_THRESHOLD}")

    def label_all(self, batch_size: int = 10, delay: float = 1.0, stop_at_target: bool = True):
        """Label all unlabeled posts, optionally stopping at ALLOWED target."""
        while True:
            # Check if we've hit the target
            if stop_at_target:
                allowed_count = self.db.get_allowed_count()
                if allowed_count >= config.TRAINING_THRESHOLD + config.TEST_SET_SIZE:
                    print(f"\n✓ Target reached! {allowed_count} ALLOWED posts collected.")
                    print(f"  Training: {config.TRAINING_THRESHOLD}")
                    print(f"  Test: {config.TEST_SET_SIZE}")
                    break

            unlabeled = self.db.get_unlabeled_posts(1)
            if not unlabeled:
                print("\nAll posts labeled!")
                allowed_count = self.db.get_allowed_count()
                if allowed_count < config.TRAINING_THRESHOLD + config.TEST_SET_SIZE:
                    remaining = (config.TRAINING_THRESHOLD + config.TEST_SET_SIZE) - allowed_count
                    print(f"⚠ Warning: Only {allowed_count} ALLOWED posts found.")
                    print(f"  Need {remaining} more. Continue scraping!")
                break

            self.label_batch(batch_size, delay)
