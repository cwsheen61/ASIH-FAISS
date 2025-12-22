#!/usr/bin/env python3
"""
Fingerprint-based similarity lookup system.
Checks both image (pHash + CLIP) and text (hash + embedding) fingerprints.
"""
import sqlite3
import json
import numpy as np
from typing import Optional, Tuple
import config

def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate Hamming distance between two hex hashes."""
    if not hash1 or not hash2:
        return 999
    xor = int(hash1, 16) ^ int(hash2, 16)
    return bin(xor).count('1')

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class FingerprintLookup:
    """
    Similarity-based lookup for moderation decisions.
    Uses image and text fingerprints to find previously moderated content.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DB_PATH

    def find_similar_content(
        self,
        image_phash: Optional[str] = None,
        clip_embedding: Optional[str] = None,
        text_hash: Optional[str] = None,
        text_embedding: Optional[str] = None,
        image_similarity_threshold: float = 0.90,
        text_similarity_threshold: float = 0.85
    ) -> Optional[Tuple[str, str, str, float]]:
        """
        Find similar previously-moderated content.

        Args:
            image_phash: Perceptual hash of image
            clip_embedding: CLIP embedding JSON string
            text_hash: SHA256 hash of text
            text_embedding: Text embedding JSON string
            image_similarity_threshold: Minimum CLIP similarity (0.90 = high confidence)
            text_similarity_threshold: Minimum text similarity (0.85 = related content)

        Returns:
            Tuple of (label, reason, match_type, confidence) or None if no match
            match_type: 'exact_image', 'similar_image', 'exact_text', 'similar_text'
        """

        db = sqlite3.connect(self.db_path)

        # Priority 1: Exact image match (pHash)
        if image_phash:
            cursor = db.execute("""
                SELECT image_phash, gemini_label, gemini_reason
                FROM posts
                WHERE image_phash IS NOT NULL
                AND gemini_label IS NOT NULL
                AND quarantined = 0
            """)

            for existing_phash, label, reason in cursor:
                distance = hamming_distance(image_phash, existing_phash)
                if distance <= config.PHASH_SIMILARITY_THRESHOLD:
                    db.close()
                    return (label, reason, 'exact_image', 1.0)

        # Priority 2: Exact text match (SHA256 hash)
        if text_hash:
            cursor = db.execute("""
                SELECT text_hash, gemini_label, gemini_reason
                FROM posts
                WHERE text_hash = ?
                AND gemini_label IS NOT NULL
                AND quarantined = 0
                LIMIT 1
            """, (text_hash,))

            row = cursor.fetchone()
            if row:
                label, reason = row[1], row[2]
                db.close()
                return (label, reason, 'exact_text', 1.0)

        # Priority 3: Similar image (CLIP ≥ threshold)
        if clip_embedding:
            cursor = db.execute("""
                SELECT clip_embedding, gemini_label, gemini_reason
                FROM posts
                WHERE clip_embedding IS NOT NULL
                AND gemini_label IS NOT NULL
                AND quarantined = 0
            """)

            new_vec = np.array(json.loads(clip_embedding))
            best_match = None
            best_similarity = image_similarity_threshold

            for existing_emb, label, reason in cursor:
                if not existing_emb:
                    continue

                existing_vec = np.array(json.loads(existing_emb))
                similarity = cosine_similarity(new_vec, existing_vec)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (label, reason, 'similar_image', similarity)

            if best_match:
                db.close()
                return best_match

        # Priority 4: Similar text (text embedding ≥ threshold)
        if text_embedding:
            cursor = db.execute("""
                SELECT text_embedding, gemini_label, gemini_reason
                FROM posts
                WHERE text_embedding IS NOT NULL
                AND gemini_label IS NOT NULL
                AND quarantined = 0
            """)

            new_vec = np.array(json.loads(text_embedding))
            best_match = None
            best_similarity = text_similarity_threshold

            for existing_emb, label, reason in cursor:
                if not existing_emb:
                    continue

                existing_vec = np.array(json.loads(existing_emb))
                similarity = cosine_similarity(new_vec, existing_vec)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (label, reason, 'similar_text', similarity)

            if best_match:
                db.close()
                return best_match

        db.close()
        return None  # No match found


# Test function
if __name__ == "__main__":
    print("="*70)
    print("TESTING FINGERPRINT LOOKUP")
    print("="*70)

    lookup = FingerprintLookup()

    # Test with a known labeled image
    db = sqlite3.connect(config.DB_PATH)
    cursor = db.execute("""
        SELECT image_phash, clip_embedding, gemini_label, post_id
        FROM posts
        WHERE image_phash IS NOT NULL
        AND clip_embedding IS NOT NULL
        AND gemini_label IS NOT NULL
        AND quarantined = 0
        LIMIT 1
    """)
    row = cursor.fetchone()
    db.close()

    if row:
        test_phash, test_clip, test_label, test_post_id = row

        print(f"\nTest post: {test_post_id}")
        print(f"Actual label: {test_label}")

        # Test exact match
        result = lookup.find_similar_content(
            image_phash=test_phash,
            clip_embedding=test_clip
        )

        if result:
            label, reason, match_type, confidence = result
            print(f"\n✓ Match found!")
            print(f"  Type: {match_type}")
            print(f"  Label: {label}")
            print(f"  Confidence: {confidence:.3f}")
        else:
            print("\n✗ No match (unexpected)")
    else:
        print("\nNo labeled images in database to test")

    print("="*70)
