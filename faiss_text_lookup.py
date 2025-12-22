#!/usr/bin/env python3
"""
Fast text similarity lookup using FAISS.
~1000x faster than brute-force comparison.
"""
import numpy as np
import faiss
import pickle
import json
from typing import Optional, Tuple
from text_fingerprint import hash_text, embed_text
import config

class FAISSTextLookup:
    """Fast text similarity lookup using FAISS index."""

    def __init__(self):
        self.index = None
        self.metadata = None
        self.loaded = False

    def load(self):
        """Load FAISS index and metadata."""
        if self.loaded:
            return

        index_path = config.DATA_DIR / "text_embeddings.faiss"
        metadata_path = config.DATA_DIR / "text_embeddings_metadata.pkl"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("FAISS index not found. Run build_faiss_index.py first.")

        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        self.loaded = True

    def find_similar(self, text: str, threshold: float = 0.85, k: int = 1) -> Optional[Tuple[str, str, float]]:
        """
        Find similar text in index using FAISS.

        Args:
            text: Text to search for
            threshold: Minimum similarity threshold (0-1)
            k: Number of nearest neighbors to retrieve

        Returns:
            Tuple of (label, reason, similarity) or None if no match above threshold
        """
        if not self.loaded:
            self.load()

        # Generate embedding
        text_emb = embed_text(text)
        if not text_emb:
            return None

        # Convert to numpy and normalize
        query_vec = np.array(json.loads(text_emb), dtype='float32').reshape(1, -1)
        faiss.normalize_L2(query_vec)

        # Search in FAISS
        similarities, indices = self.index.search(query_vec, k)

        # Check if best match exceeds threshold
        if similarities[0][0] >= threshold:
            idx = indices[0][0]
            meta = self.metadata[idx]
            return (meta['label'], meta['reason'], float(similarities[0][0]))

        return None

    def find_text_match(self, text: str) -> Optional[Tuple[str, str, str, float]]:
        """
        Complete text matching: exact hash + FAISS semantic search.

        Returns:
            Tuple of (label, reason, match_type, confidence) or None
        """
        # Step 1: Exact hash match (instant)
        text_hash = hash_text(text)
        if text_hash:
            # Would need to add hash lookup to FAISS or keep small hash table
            # For now, skip exact match in this implementation
            pass

        # Step 2: FAISS semantic search
        result = self.find_similar(text, threshold=0.85)
        if result:
            label, reason, similarity = result
            return (label, reason, 'semantic', similarity)

        return None

    def get_stats(self):
        """Get index statistics."""
        if not self.loaded:
            self.load()

        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.index.d,
        }


# Test
if __name__ == "__main__":
    print("Testing FAISS text lookup...")

    lookup = FAISSTextLookup()
    lookup.load()

    stats = lookup.get_stats()
    print(f"Loaded index with {stats['total_vectors']:,} vectors")
    print(f"Dimension: {stats['dimension']}")

    # Test search
    test_text = "I hate all Jews and they should be eliminated"

    import time
    start = time.time()
    result = lookup.find_similar(test_text, threshold=0.85)
    elapsed = time.time() - start

    if result:
        label, reason, similarity = result
        print(f"\nTest search: {elapsed*1000:.1f}ms")
        print(f"Found match: {label} (similarity: {similarity:.3f})")
        print(f"Reason: {reason[:100]}")
    else:
        print(f"\nTest search: {elapsed*1000:.1f}ms")
        print("No match found")
