#!/usr/bin/env python3
"""
Fast moderation pipeline using CLIP for image triage, then text confirmation.

Workflow:
1. CLIP image classifier: "I've seen this image before, probably good/bad"
2. Text analysis: Confirm or override the image prediction
"""
import re
import json
import sqlite3
from typing import Tuple, Optional
from pathlib import Path
import config
from clip_classifier import CLIPClassifier

class FastModerationPipeline:
    """Fast two-stage moderation: image triage → text confirmation."""

    def __init__(self):
        self.clip_classifier = CLIPClassifier()
        self.db_path = config.DB_PATH

        # Simple text patterns for known hate speech
        self.hate_patterns = [
            # Racial slurs
            r'\bn+i+g+e*r+s?\b',
            r'\bn+i+g+a+s?\b',
            r'\bk+i+k+e+s?\b',
            r'\bc+o+o+n+s?\b',
            r'\bs+p+i+c+s?\b',
            r'\bg+o+o+k+s?\b',
            r'\bc+h+i+n+k+s?\b',

            # Homophobic slurs
            r'\bf+a+g+o*t+s?\b',
            r'\bf+a+g+s?\b',
            r'\bd+y+k+e+s?\b',
            r'\bt+r+a+n+y+s?\b',

            # Antisemitic markers
            r'\(\(\(.+?\)\)\)',  # Triple parentheses
            r'\bj+e+w+s? control',
            r'\bj+e+w+i+s+h+ conspiracy',

            # General hate
            r'\bsubhumans?\b',
            r'\bwhite genocide\b',
            r'\brace mixing\b',
            r'\bmud ?shark',
            r'\bcoal ?burn',
        ]

        self.hate_regex = re.compile('|'.join(self.hate_patterns), re.IGNORECASE)

    def check_text_for_hate(self, text: str) -> Tuple[bool, str]:
        """
        Quick text check for obvious hate speech.

        Returns:
            (is_hateful, reason)
        """
        if not text:
            return False, ""

        # Remove HTML tags for cleaner matching
        clean_text = re.sub(r'<[^>]+>', '', text)

        # Check for hate patterns
        match = self.hate_regex.search(clean_text)
        if match:
            return True, f"Contains hate speech pattern: {match.group(0)}"

        # Check for excessive caps (often used in hate speech)
        if len(clean_text) > 20:
            caps_ratio = sum(1 for c in clean_text if c.isupper()) / len(clean_text)
            if caps_ratio > 0.7:
                # High caps + certain keywords
                if any(word in clean_text.upper() for word in ['HATE', 'KILL', 'DIE', 'DESTROY']):
                    return True, "All-caps hate speech"

        return False, ""

    def moderate_post(self, post_id: int = None,
                     clip_embedding: str = None,
                     text: str = None,
                     image_path: str = None) -> dict:
        """
        Moderate a post using two-stage pipeline.

        Args:
            post_id: Database post ID (if checking existing post)
            clip_embedding: CLIP embedding JSON (if new post)
            text: Text content
            image_path: Path to image file

        Returns:
            {
                'final_label': 'ALLOWED' or 'DENIED',
                'image_prediction': 'ALLOWED/DENIED',
                'image_confidence': float,
                'text_flagged': bool,
                'text_reason': str,
                'decision_path': str
            }
        """
        result = {
            'final_label': 'ALLOWED',
            'image_prediction': None,
            'image_confidence': 0.0,
            'text_flagged': False,
            'text_reason': '',
            'decision_path': ''
        }

        # Stage 1: Image triage (if we have an image)
        if clip_embedding:
            image_label, image_conf = self.clip_classifier.predict(clip_embedding)
            result['image_prediction'] = image_label
            result['image_confidence'] = image_conf

            # High-confidence bad image → likely DENIED
            if image_label == 'DENIED' and image_conf > 0.70:
                result['decision_path'] = 'Image strongly suggests DENIED (>70% conf)'
                result['final_label'] = 'DENIED'

                # Still check text to confirm
                if text:
                    text_flagged, text_reason = self.check_text_for_hate(text)
                    result['text_flagged'] = text_flagged
                    result['text_reason'] = text_reason

                    if text_flagged:
                        result['decision_path'] += ' + Text confirms hate speech → DENIED'
                    else:
                        result['decision_path'] += ' + Text clean, but image is bad → DENIED'

                return result

            # Uncertain or low-confidence → defer to text
            elif image_conf < 0.65:
                result['decision_path'] = 'Image uncertain (<65% conf), checking text'

            # High-confidence good image
            else:
                result['decision_path'] = 'Image suggests ALLOWED, checking text to confirm'

        # Stage 2: Text confirmation
        if text:
            text_flagged, text_reason = self.check_text_for_hate(text)
            result['text_flagged'] = text_flagged
            result['text_reason'] = text_reason

            if text_flagged:
                result['final_label'] = 'DENIED'
                result['decision_path'] += f' → Text contains hate: {text_reason} → DENIED'
            else:
                # Text is clean
                if result['image_prediction'] == 'DENIED':
                    # Image suggested bad, but text is clean
                    result['final_label'] = 'UNCERTAIN'  # Escalate to manual review
                    result['decision_path'] += ' → Image bad but text clean → UNCERTAIN (escalate)'
                else:
                    result['final_label'] = 'ALLOWED'
                    result['decision_path'] += ' → Text clean → ALLOWED'
        else:
            # No text, go with image prediction
            if result['image_prediction']:
                result['final_label'] = result['image_prediction']
                result['decision_path'] += f' → No text, image only → {result["final_label"]}'

        return result

    def batch_moderate(self, limit: int = 100, save_to_db: bool = False):
        """
        Moderate a batch of unlabeled posts.

        Args:
            limit: Number of posts to process
            save_to_db: Whether to save predictions back to database
        """
        print("="*70)
        print("FAST MODERATION PIPELINE")
        print("="*70)
        print(f"Processing up to {limit} posts...\n")

        # Get unlabeled posts
        db = sqlite3.connect(self.db_path)
        cursor = db.execute("""
            SELECT id, post_id, source, clip_embedding, content_text, encoded_path
            FROM posts
            WHERE clip_embedding IS NOT NULL
            AND gemini_label IS NULL
            AND quarantined = 0
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))

        posts = cursor.fetchall()
        db.close()

        if not posts:
            print("No unlabeled posts found")
            return

        print(f"Found {len(posts)} posts to moderate\n")

        stats = {
            'ALLOWED': 0,
            'DENIED': 0,
            'UNCERTAIN': 0
        }

        results = []

        for db_id, post_id, source, clip_emb, text, image_path in posts:
            result = self.moderate_post(
                post_id=db_id,
                clip_embedding=clip_emb,
                text=text,
                image_path=image_path
            )

            stats[result['final_label']] += 1
            results.append((db_id, post_id, result))

            # Print summary
            emoji = "✓" if result['final_label'] == 'ALLOWED' else "⚠" if result['final_label'] == 'DENIED' else "?"
            print(f"{emoji} Post {post_id}: {result['final_label']}")
            print(f"   {result['decision_path']}")
            if result['text_flagged']:
                print(f"   Text: {result['text_reason']}")
            print()

        # Summary
        print("="*70)
        print("MODERATION SUMMARY")
        print("="*70)
        print(f"Total processed: {len(posts)}")
        print(f"  ALLOWED:   {stats['ALLOWED']} ({stats['ALLOWED']/len(posts)*100:.1f}%)")
        print(f"  DENIED:    {stats['DENIED']} ({stats['DENIED']/len(posts)*100:.1f}%)")
        print(f"  UNCERTAIN: {stats['UNCERTAIN']} ({stats['UNCERTAIN']/len(posts)*100:.1f}%)")
        print("="*70)

        return results


def main():
    """Test the fast moderation pipeline."""
    pipeline = FastModerationPipeline()

    # Test on recent posts
    results = pipeline.batch_moderate(limit=35, save_to_db=False)

    print("\nPipeline ready for production use!")
    print("- Fast CLIP triage for images")
    print("- Text pattern matching for confirmation")
    print("- Escalates uncertain cases for manual review")


if __name__ == "__main__":
    main()
