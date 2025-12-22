"""Database management for scraped content."""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import config


def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate Hamming distance between two hex hash strings."""
    if not hash1 or not hash2:
        return 999  # Return high value if either is None
    try:
        # Convert hex strings to integers and XOR them
        h1 = int(hash1, 16)
        h2 = int(hash2, 16)
        xor = h1 ^ h2
        # Count the number of 1s in the binary representation
        return bin(xor).count('1')
    except (ValueError, TypeError):
        return 999


def cosine_similarity(embedding1: str, embedding2: str) -> float:
    """Calculate cosine similarity between two CLIP embeddings."""
    if not embedding1 or not embedding2:
        return 0.0
    try:
        vec1 = np.array(json.loads(embedding1))
        vec2 = np.array(json.loads(embedding2))
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    except (ValueError, TypeError, json.JSONDecodeError):
        return 0.0


class Database:
    """Manages storage of scraped content metadata."""

    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    post_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    content_text TEXT,
                    post_type TEXT,
                    quarantined BOOLEAN DEFAULT 0,
                    has_image BOOLEAN,
                    image_filename TEXT,
                    encoded_path TEXT,
                    image_phash TEXT,
                    clip_embedding TEXT,
                    scraped_at TIMESTAMP,
                    gemini_label TEXT,
                    gemini_reason TEXT,
                    gemini_checked_at TIMESTAMP,
                    dataset_split TEXT,
                    UNIQUE(source, post_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_gemini_label
                ON posts(gemini_label)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_image_phash
                ON posts(image_phash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dataset_split
                ON posts(dataset_split)
            """)
            conn.commit()

    def add_post(self, source: str, post_id: str, thread_id: str,
                 content_text: Optional[str], post_type: str,
                 quarantined: bool = False, has_image: bool = False,
                 image_filename: Optional[str] = None, encoded_path: Optional[str] = None,
                 image_phash: Optional[str] = None, clip_embedding: Optional[str] = None,
                 ocr_text: Optional[str] = None, text_hash: Optional[str] = None,
                 text_embedding: Optional[str] = None) -> int:
        """Add a new post to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT OR IGNORE INTO posts
                (source, post_id, thread_id, content_text, post_type, quarantined,
                 has_image, image_filename, encoded_path, image_phash, clip_embedding,
                 ocr_text, text_hash, text_embedding, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (source, post_id, thread_id, content_text, post_type, quarantined,
                  has_image, image_filename, encoded_path, image_phash, clip_embedding,
                  ocr_text, text_hash, text_embedding, datetime.now()))
            conn.commit()
            return cursor.lastrowid

    def update_gemini_label(self, post_id: int, label: str, reason: str = None):
        """Update Gemini classification label for a post."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE posts
                SET gemini_label = ?, gemini_reason = ?, gemini_checked_at = ?
                WHERE id = ?
            """, (label, reason, datetime.now(), post_id))
            conn.commit()

    def get_unlabeled_posts(self, limit: int = 100) -> List[Tuple]:
        """Get posts that haven't been labeled by Gemini yet."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, content_text, encoded_path, has_image, image_phash
                FROM posts
                WHERE gemini_label IS NULL AND quarantined = 0
                LIMIT ?
            """, (limit,))
            return cursor.fetchall()

    def get_label_by_phash(self, phash: str) -> Optional[Tuple[str, str]]:
        """Get existing label for a similar image using perceptual hash."""
        if not phash:
            return None
        with sqlite3.connect(self.db_path) as conn:
            # Get all labeled posts with phash
            cursor = conn.execute("""
                SELECT image_phash, gemini_label, gemini_reason
                FROM posts
                WHERE image_phash IS NOT NULL AND gemini_label IS NOT NULL
            """)

            # Find the most similar hash below threshold
            for existing_phash, label, reason in cursor:
                distance = hamming_distance(phash, existing_phash)
                if distance <= config.PHASH_SIMILARITY_THRESHOLD:
                    return (label, reason)

            return None

    def is_duplicate_image(self, phash: str, clip_embedding: Optional[str] = None) -> bool:
        """Check if an image with similar phash or CLIP embedding already exists."""
        if not phash and not clip_embedding:
            return False

        with sqlite3.connect(self.db_path) as conn:
            # Fast check: pHash
            if phash:
                cursor = conn.execute("""
                    SELECT image_phash
                    FROM posts
                    WHERE image_phash IS NOT NULL
                """)

                for (existing_phash,) in cursor:
                    distance = hamming_distance(phash, existing_phash)
                    if distance <= config.PHASH_SIMILARITY_THRESHOLD:
                        return True

            # Semantic check: CLIP (if phash didn't match)
            if clip_embedding:
                cursor = conn.execute("""
                    SELECT clip_embedding
                    FROM posts
                    WHERE clip_embedding IS NOT NULL
                """)

                for (existing_embedding,) in cursor:
                    similarity = cosine_similarity(clip_embedding, existing_embedding)
                    if similarity >= config.CLIP_SIMILARITY_THRESHOLD:
                        return True

            return False

    def get_label_counts(self) -> dict:
        """Get counts of labels."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT gemini_label, COUNT(*)
                FROM posts
                WHERE gemini_label IS NOT NULL
                GROUP BY gemini_label
            """)
            return dict(cursor.fetchall())

    def assign_dataset_splits(self, test_size: int = config.TEST_SET_SIZE):
        """Assign posts to train/test splits."""
        with sqlite3.connect(self.db_path) as conn:
            # Assign test set randomly
            conn.execute("""
                UPDATE posts
                SET dataset_split = 'test'
                WHERE id IN (
                    SELECT id FROM posts
                    WHERE gemini_label IS NOT NULL
                    AND dataset_split IS NULL
                    ORDER BY RANDOM()
                    LIMIT ?
                )
            """, (test_size,))

            # Rest goes to training
            conn.execute("""
                UPDATE posts
                SET dataset_split = 'train'
                WHERE gemini_label IS NOT NULL
                AND dataset_split IS NULL
            """)
            conn.commit()

    def get_training_data(self) -> List[Tuple]:
        """Get all training data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT encoded_path, gemini_label, content_text, has_image
                FROM posts
                WHERE dataset_split = 'train'
            """)
            return cursor.fetchall()

    def get_test_data(self) -> List[Tuple]:
        """Get all test data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT encoded_path, gemini_label, content_text, has_image
                FROM posts
                WHERE dataset_split = 'test'
            """)
            return cursor.fetchall()

    def get_allowed_count(self, source: Optional[str] = None) -> int:
        """Get count of ALLOWED posts, optionally filtered by source."""
        with sqlite3.connect(self.db_path) as conn:
            if source:
                result = conn.execute("""
                    SELECT COUNT(*) FROM posts
                    WHERE gemini_label = 'ALLOWED' AND source = ?
                """, (source,)).fetchone()[0]
            else:
                result = conn.execute("""
                    SELECT COUNT(*) FROM posts
                    WHERE gemini_label = 'ALLOWED'
                """).fetchone()[0]
            return result

    def get_stats(self) -> dict:
        """Get overall statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
            quarantined = conn.execute(
                "SELECT COUNT(*) FROM posts WHERE quarantined = 1"
            ).fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM posts WHERE quarantined = 0"
            ).fetchone()[0]
            labeled = conn.execute(
                "SELECT COUNT(*) FROM posts WHERE gemini_label IS NOT NULL"
            ).fetchone()[0]
            allowed = conn.execute(
                "SELECT COUNT(*) FROM posts WHERE gemini_label = 'ALLOWED'"
            ).fetchone()[0]
            denied = conn.execute(
                "SELECT COUNT(*) FROM posts WHERE gemini_label = 'DENIED'"
            ).fetchone()[0]
            train = conn.execute(
                "SELECT COUNT(*) FROM posts WHERE dataset_split = 'train'"
            ).fetchone()[0]
            test = conn.execute(
                "SELECT COUNT(*) FROM posts WHERE dataset_split = 'test'"
            ).fetchone()[0]
            unique_images = conn.execute(
                "SELECT COUNT(DISTINCT image_phash) FROM posts WHERE image_phash IS NOT NULL AND quarantined = 0"
            ).fetchone()[0]

            # Post type breakdown
            type_breakdown = {}
            cursor = conn.execute("""
                SELECT post_type, COUNT(*) FROM posts
                GROUP BY post_type
            """)
            for post_type, count in cursor:
                if post_type:
                    type_breakdown[post_type] = count

            return {
                "total_posts": total,
                "active_posts": active,
                "quarantined_posts": quarantined,
                "labeled_posts": labeled,
                "allowed_posts": allowed,
                "denied_posts": denied,
                "training_posts": train,
                "test_posts": test,
                "unique_images": unique_images,
                "type_breakdown": type_breakdown
            }

    def get_source_breakdown(self) -> dict:
        """Get breakdown of ALLOWED posts by source."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT source, COUNT(*) as count
                FROM posts
                WHERE gemini_label = 'ALLOWED'
                GROUP BY source
            """)
            return dict(cursor.fetchall())
