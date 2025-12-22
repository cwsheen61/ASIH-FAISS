#!/usr/bin/env python3
"""
Text fingerprinting utilities for encoded hate detection.
Generates hashes and embeddings without storing plain text.
"""
import hashlib
import json
from typing import Optional, Tuple
from sentence_transformers import SentenceTransformer
import easyocr
import numpy as np
from io import BytesIO

# Global models (loaded once)
_text_model = None
_ocr_reader = None

def get_text_encoder():
    """Lazy load sentence transformer model."""
    global _text_model
    if _text_model is None:
        print("Loading text encoder (one-time setup)...")
        _text_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, efficient
        print("Text encoder loaded successfully")
    return _text_model

def get_ocr_reader():
    """Lazy load OCR reader."""
    global _ocr_reader
    if _ocr_reader is None:
        print("Loading OCR reader (one-time setup)...")
        _ocr_reader = easyocr.Reader(['en'], gpu=True)
        print("OCR reader loaded successfully")
    return _ocr_reader

def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from image using OCR.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Extracted text string (empty if no text found)
    """
    try:
        reader = get_ocr_reader()
        result = reader.readtext(image_bytes, detail=0)  # Just text, no bounding boxes
        return ' '.join(result)
    except Exception as e:
        print(f"OCR failed: {e}")
        return ""

def hash_text(text: str) -> str:
    """
    Generate SHA256 hash of text for exact duplicate detection.

    Args:
        text: Plain text string

    Returns:
        Hex string hash
    """
    if not text:
        return ""
    # Normalize: lowercase, strip whitespace
    normalized = text.lower().strip()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

def embed_text(text: str) -> Optional[str]:
    """
    Generate semantic embedding for text similarity matching.

    Args:
        text: Plain text string

    Returns:
        JSON string of embedding vector (384 dimensions)
    """
    if not text:
        return None

    try:
        model = get_text_encoder()
        embedding = model.encode(text)
        return json.dumps(embedding.tolist())
    except Exception as e:
        print(f"Text embedding failed: {e}")
        return None

def generate_text_fingerprint(post_text: str, ocr_text: str) -> Tuple[str, Optional[str]]:
    """
    Generate complete text fingerprint from post caption + OCR.

    Args:
        post_text: Post caption/comment text
        ocr_text: Text extracted from image via OCR

    Returns:
        Tuple of (text_hash, text_embedding)
    """
    # Combine all text sources
    full_text = f"{post_text} {ocr_text}".strip()

    if not full_text:
        return "", None

    # Generate fingerprints
    text_hash = hash_text(full_text)
    text_embedding = embed_text(full_text)

    return text_hash, text_embedding

def text_similarity(emb1_json: str, emb2_json: str) -> float:
    """
    Calculate cosine similarity between two text embeddings.

    Args:
        emb1_json: First embedding as JSON string
        emb2_json: Second embedding as JSON string

    Returns:
        Cosine similarity (0-1)
    """
    vec1 = np.array(json.loads(emb1_json))
    vec2 = np.array(json.loads(emb2_json))
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Test function
if __name__ == "__main__":
    # Test text fingerprinting
    test_text = "This is hateful content targeting minorities"

    print("Testing text fingerprinting...")
    print(f"Input: {test_text}")

    text_hash = hash_text(test_text)
    print(f"Hash: {text_hash[:16]}...")

    text_emb = embed_text(test_text)
    print(f"Embedding: {len(json.loads(text_emb))} dimensions")

    # Test similarity
    similar_text = "This is hateful content targeting minorities!"  # Same with punctuation
    similar_emb = embed_text(similar_text)
    sim = text_similarity(text_emb, similar_emb)
    print(f"Similarity to similar text: {sim:.3f}")

    different_text = "This is a nice cat picture"
    different_emb = embed_text(different_text)
    sim2 = text_similarity(text_emb, different_emb)
    print(f"Similarity to different text: {sim2:.3f}")
