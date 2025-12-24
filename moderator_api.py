#!/usr/bin/env python3
"""
Content Moderation API Service
Fast Unix socket API for Gocial integration
"""
import os
import sys
import time
import json
import base64
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import faiss
import pickle

# Local imports
from text_fingerprint import hash_text, get_text_encoder, generate_text_fingerprint
from database import Database
import config

# Initialize FastAPI
app = FastAPI(title="Content Moderator", version="1.0.0")


@dataclass
class ModerationResult:
    """Result from content moderation check."""
    status: str  # "ALLOWED" or "DENIED"
    confidence: float  # 0-1 similarity score
    match_type: Optional[str] = None  # "exact_hash", "semantic", or None
    reason: Optional[str] = None  # Why it was denied
    lookup_time_ms: float = 0.0


class TextCheckRequest(BaseModel):
    text: str


class ImageCheckRequest(BaseModel):
    image_base64: str  # Base64 encoded image bytes


class ContentCheckResponse(BaseModel):
    status: str
    confidence: float
    match_type: Optional[str]
    reason: Optional[str]
    lookup_time_ms: float


class ModeratorService:
    """Fast content moderation using FAISS indices."""

    def __init__(self):
        self.text_encoder = None
        self.text_index = None
        self.text_metadata = None
        self.image_index = None
        self.image_metadata = None
        self.loaded = False
        self.db = Database()
        self.ollama_available = self._check_ollama()

    def load(self):
        """Load FAISS indices and models (one-time setup)."""
        if self.loaded:
            return

        print("Loading moderation models...")
        start = time.time()

        # Load text encoder
        print("  - Text encoder")
        self.text_encoder = get_text_encoder()

        # Load text FAISS index
        text_index_path = config.DATA_DIR / "text_embeddings.faiss"
        text_metadata_path = config.DATA_DIR / "text_embeddings_metadata.pkl"

        if text_index_path.exists() and text_metadata_path.exists():
            print(f"  - Text FAISS index")
            self.text_index = faiss.read_index(str(text_index_path))
            with open(text_metadata_path, 'rb') as f:
                self.text_metadata = pickle.load(f)
            print(f"    → {self.text_index.ntotal:,} text vectors loaded")
        else:
            print("  ⚠ No text FAISS index found")

        # Load image FAISS index
        image_index_path = config.DATA_DIR / "image_embeddings.faiss"
        image_metadata_path = config.DATA_DIR / "image_embeddings_metadata.pkl"

        if image_index_path.exists() and image_metadata_path.exists():
            print(f"  - Image FAISS index")
            self.image_index = faiss.read_index(str(image_index_path))
            with open(image_metadata_path, 'rb') as f:
                self.image_metadata = pickle.load(f)
            print(f"    → {self.image_index.ntotal:,} image vectors loaded")
        else:
            print("  ⚠ No image FAISS index found")

        # Load CLIP model (lazy load - only if needed)
        # We'll load this on first image check

        elapsed = time.time() - start
        print(f"✓ Models loaded in {elapsed:.1f}s\n")
        self.loaded = True

    def _check_ollama(self) -> bool:
        """Check if Ollama is available for fallback moderation."""
        try:
            import requests
            response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def _mark_indices_dirty(self):
        """Mark indices as needing rebuild."""
        dirty_file = config.DATA_DIR / ".indices_dirty"
        dirty_file.touch()

    def _fallback_moderate_text(self, text: str) -> tuple[str, str]:
        """
        Fallback text moderation using Ollama when FAISS doesn't have a match.

        Returns:
            (verdict, reason) tuple - ("ALLOWED" or "DENIED", reason string)
        """
        if not self.ollama_available:
            return ("ALLOWED", "Novel content (no fallback moderation available)")

        try:
            import requests

            prompt = f"""Analyze this text for hate speech, bigotry, or policy violations.

Text: "{text}"

Respond in JSON format:
{{"classification": "ALLOWED" or "DENIED", "reason": "brief explanation"}}"""

            response = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '{}')

                try:
                    parsed = json.loads(response_text)
                    classification = parsed.get('classification', 'ALLOWED').upper()
                    reason = parsed.get('reason', 'Content moderated by Ollama')

                    if classification in ['ALLOWED', 'DENIED']:
                        return (classification, reason)
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            print(f"Ollama fallback error: {e}")

        # Default to ALLOWED if moderation fails
        return ("ALLOWED", "Novel content (moderation failed)")

    def _save_text_to_db(self, text: str, text_hash: str, text_embedding_json: str, verdict: str, reason: str):
        """Save moderated text content to database for future lookups."""
        try:
            import sqlite3
            from datetime import datetime

            # Do everything in one transaction
            with sqlite3.connect(self.db.db_path) as conn:
                # Insert post
                cursor = conn.execute("""
                    INSERT OR IGNORE INTO posts
                    (source, post_id, thread_id, content_text, post_type, quarantined,
                     text_hash, text_embedding, scraped_at, gemini_label, gemini_reason, first_seen_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, ("gocial/production", f"gocial_{text_hash[:16]}", "0", text[:1000], 'text',
                      False, text_hash, text_embedding_json, datetime.now(), verdict, reason))

                db_id = cursor.lastrowid
                conn.commit()

            # Mark indices as needing rebuild
            self._mark_indices_dirty()

            print(f"✓ Saved novel text content to DB: {verdict} (id={db_id})")

        except Exception as e:
            print(f"Error saving text to DB: {e}")

    def _record_cache_hit(self, db_id: int):
        """Record that a fingerprint was matched (cache hit)."""
        try:
            import sqlite3
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute(
                    "UPDATE posts SET last_hit_at = CURRENT_TIMESTAMP, hit_count = hit_count + 1 WHERE id = ?",
                    (db_id,)
                )
                conn.commit()
        except Exception as e:
            print(f"Error recording cache hit: {e}")

    def _save_image_to_db(self, image_bytes: bytes, phash: str, clip_embedding: str, verdict: str, reason: str):
        """Save moderated image content to database for future lookups."""
        try:
            import sqlite3
            from datetime import datetime

            # Encode and save image
            encoded_filename = f"gocial_{phash}.jpg.b64"
            encoded_path = config.ENCODED_DIR / encoded_filename

            encoded_data = base64.b64encode(image_bytes)
            with open(encoded_path, 'wb') as f:
                f.write(encoded_data)

            # Do everything in one transaction
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.execute("""
                    INSERT OR IGNORE INTO posts
                    (source, post_id, thread_id, content_text, post_type, quarantined,
                     has_image, image_filename, encoded_path, image_phash, clip_embedding,
                     scraped_at, gemini_label, gemini_reason, first_seen_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, ("gocial/production", f"gocial_{phash}", "0", "", 'image', False,
                      True, encoded_filename, str(encoded_path), phash, clip_embedding,
                      datetime.now(), verdict, reason))

                db_id = cursor.lastrowid
                conn.commit()

            # Mark indices as needing rebuild
            self._mark_indices_dirty()

            print(f"✓ Saved novel image content to DB: {verdict} (id={db_id})")

        except Exception as e:
            print(f"Error saving image to DB: {e}")

    def check_text(self, text: str, threshold: float = 0.85) -> ModerationResult:
        """
        Check text against fingerprint database.

        Args:
            text: Text content to check
            threshold: Similarity threshold (0-1)

        Returns:
            ModerationResult with verdict
        """
        start = time.time()

        if not self.loaded:
            self.load()

        if not self.text_index:
            return ModerationResult(
                status="ALLOWED",
                confidence=0.0,
                reason="No text index available",
                lookup_time_ms=(time.time() - start) * 1000
            )

        # Step 1: Check exact hash (instant)
        text_hash = hash_text(text)
        if text_hash:
            # Would need hash lookup table - skip for now
            pass

        # Step 2: FAISS semantic search
        try:
            emb = self.text_encoder.encode(text)
            query_vec = np.array(emb, dtype='float32').reshape(1, -1)
            faiss.normalize_L2(query_vec)

            similarities, indices = self.text_index.search(query_vec, 1)

            if similarities[0][0] >= threshold:
                idx = indices[0][0]
                meta = self.text_metadata[idx]

                # Record cache hit
                self._record_cache_hit(meta['id'])

                lookup_time = (time.time() - start) * 1000

                return ModerationResult(
                    status=meta['label'],
                    confidence=float(similarities[0][0]),
                    match_type='semantic',
                    reason=meta['reason'],
                    lookup_time_ms=lookup_time
                )
        except Exception as e:
            print(f"Error in text check: {e}")

        # No match - content is novel, use fallback moderation
        # Generate fingerprint for saving (no OCR text for pure text content)
        text_hash, text_embedding_json = generate_text_fingerprint(text, "")

        # Call Ollama for moderation
        verdict, reason = self._fallback_moderate_text(text)

        # Save to database for future lookups
        self._save_text_to_db(text, text_hash, text_embedding_json, verdict, reason)

        lookup_time = (time.time() - start) * 1000

        return ModerationResult(
            status=verdict,
            confidence=1.0 if verdict == "DENIED" else 0.0,
            match_type=None,  # Novel content
            reason=reason,
            lookup_time_ms=lookup_time
        )

    def check_image(self, image_bytes: bytes, threshold: float = 0.90) -> ModerationResult:
        """
        Check image against fingerprint database.

        Args:
            image_bytes: Raw image bytes
            threshold: Similarity threshold (0-1)

        Returns:
            ModerationResult with verdict
        """
        start = time.time()

        if not self.loaded:
            self.load()

        if not self.image_index:
            return ModerationResult(
                status="ALLOWED",
                confidence=0.0,
                reason="No image index available",
                lookup_time_ms=(time.time() - start) * 1000
            )

        # FAISS CLIP semantic search
        try:
            # Generate CLIP embedding
            from clip_encoder import generate_clip_embedding
            clip_emb = generate_clip_embedding(image_bytes)

            if clip_emb:
                clip_vec = np.array(json.loads(clip_emb), dtype='float32').reshape(1, -1)
                faiss.normalize_L2(clip_vec)

                similarities, indices = self.image_index.search(clip_vec, 1)

                if similarities[0][0] >= threshold:
                    idx = indices[0][0]
                    meta = self.image_metadata[idx]

                    # Record cache hit
                    self._record_cache_hit(meta['id'])

                    lookup_time = (time.time() - start) * 1000

                    return ModerationResult(
                        status=meta['label'],
                        confidence=float(similarities[0][0]),
                        match_type='semantic',
                        reason=meta['reason'],
                        lookup_time_ms=lookup_time
                    )
        except Exception as e:
            print(f"Error in image check: {e}")

        # No match - content is novel
        # For images, default to ALLOWED but save to database for future training
        # (Gemini fallback would be expensive - handle in batch labeling instead)

        try:
            from PIL import Image
            import imagehash
            from io import BytesIO

            # Generate pHash
            img = Image.open(BytesIO(image_bytes))
            phash = str(imagehash.phash(img))

            # We already have CLIP embedding from above
            if clip_emb:
                # Save novel image to database with ALLOWED verdict
                # It will be labeled properly in next batch labeling run
                self._save_image_to_db(image_bytes, phash, clip_emb, "ALLOWED", "Novel content (needs manual review)")
        except Exception as e:
            print(f"Error saving novel image: {e}")

        lookup_time = (time.time() - start) * 1000

        return ModerationResult(
            status="ALLOWED",  # Fail open for novel images
            confidence=0.0,
            match_type=None,
            reason="Novel content (saved for review)",
            lookup_time_ms=lookup_time
        )


# Global moderator instance
moderator = ModeratorService()


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    moderator.load()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "text_index_loaded": moderator.text_index is not None,
        "image_index_loaded": moderator.image_index is not None,
        "text_vectors": moderator.text_index.ntotal if moderator.text_index else 0,
        "image_vectors": moderator.image_index.ntotal if moderator.image_index else 0,
    }


@app.post("/check/text", response_model=ContentCheckResponse)
async def check_text(request: TextCheckRequest):
    """
    Check text content for policy violations.

    Returns verdict in ~5ms for cached content.
    """
    try:
        result = moderator.check_text(request.text)
        return ContentCheckResponse(**asdict(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/check/image", response_model=ContentCheckResponse)
async def check_image(request: ImageCheckRequest):
    """
    Check image content for policy violations.

    Expects base64 encoded image.
    Returns verdict in ~5-10ms for cached content.
    """
    try:
        # Decode base64
        image_bytes = base64.b64decode(request.image_base64)

        result = moderator.check_image(image_bytes)
        return ContentCheckResponse(**asdict(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get moderation statistics."""
    return {
        "text_index": {
            "total_vectors": moderator.text_index.ntotal if moderator.text_index else 0,
            "dimension": moderator.text_index.d if moderator.text_index else 0,
        },
        "image_index": {
            "total_vectors": moderator.image_index.ntotal if moderator.image_index else 0,
            "dimension": moderator.image_index.d if moderator.image_index else 0,
        }
    }


def main():
    """Run the API server."""
    import argparse

    parser = argparse.ArgumentParser(description="Content Moderation API")
    parser.add_argument(
        "--socket",
        type=str,
        default="/tmp/moderator.sock",
        help="Unix socket path"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="TCP host (if not using Unix socket)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port (if not using Unix socket)"
    )

    args = parser.parse_args()

    if args.host:
        # TCP mode
        print(f"Starting API server on {args.host}:{args.port}")
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info"
        )
    else:
        # Unix socket mode (default)
        socket_path = args.socket

        # Remove existing socket
        if os.path.exists(socket_path):
            os.remove(socket_path)

        print(f"Starting API server on Unix socket: {socket_path}")
        uvicorn.run(
            app,
            uds=socket_path,
            log_level="info"
        )


if __name__ == "__main__":
    main()
