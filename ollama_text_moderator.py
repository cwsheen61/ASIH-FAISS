#!/usr/bin/env python3
"""
Ollama-based text moderation for local, fast, free hate speech detection.
"""
import requests
import json
from typing import Optional, Tuple

class OllamaTextModerator:
    """Local text moderation using Ollama content-moderator model."""

    def __init__(self, model: str = "content-moderator", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def classify_text(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Classify text as DENIED or ALLOWED.

        Args:
            text: Text content to moderate

        Returns:
            Tuple of (label, reason) or None if failed
        """
        if not text or len(text.strip()) < 3:
            return None

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
                        "temperature": 0.1,
                    }
                },
                timeout=30
            )

            if response.status_code != 200:
                return None

            result = response.json()
            text_response = result.get('response', '')

            # Extract JSON from response
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


# Test
if __name__ == "__main__":
    moderator = OllamaTextModerator()

    test_cases = [
        "This is a normal comment about politics",
        "I hate all Jews and they should be eliminated",
        "The weather is nice today",
        "F*cking n*ggers ruining everything",
    ]

    print("Testing Ollama text moderator:")
    for text in test_cases:
        result = moderator.classify_text(text)
        if result:
            label, reason = result
            print(f"\nText: {text[:50]}...")
            print(f"  Result: {label}")
            print(f"  Reason: {reason[:80]}")
        else:
            print(f"\nText: {text[:50]}...")
            print(f"  Result: FAILED")
