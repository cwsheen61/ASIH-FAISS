#!/usr/bin/env python3
"""
Build FAISS index for fast text similarity search.
Converts text embeddings from database into FAISS index for ~1000x speedup.
"""
import sqlite3
import json
import numpy as np
import faiss
import pickle
import config

def build_faiss_index():
    """Build FAISS index from text embeddings in database."""
    print("="*70)
    print("BUILDING FAISS INDEX FOR TEXT SIMILARITY")
    print("="*70)

    # Load all text embeddings from database
    print("\nLoading text embeddings from database...")
    db = sqlite3.connect(config.DB_PATH)
    cursor = db.execute("""
        SELECT id, text_embedding, gemini_label, gemini_reason
        FROM posts
        WHERE text_embedding IS NOT NULL
        AND gemini_label IS NOT NULL
    """)

    embeddings = []
    metadata = []

    for db_id, emb_json, label, reason in cursor:
        emb = np.array(json.loads(emb_json), dtype='float32')
        embeddings.append(emb)
        metadata.append({
            'id': db_id,
            'label': label,
            'reason': reason
        })

    db.close()

    if not embeddings:
        print("No embeddings found in database!")
        return

    print(f"✓ Loaded {len(embeddings)} text embeddings")

    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype='float32')
    dimension = embeddings_array.shape[1]

    print(f"✓ Embedding dimension: {dimension}")

    # Build FAISS index
    # Using IndexFlatIP (inner product) which is equivalent to cosine similarity for normalized vectors
    print("\nBuilding FAISS index...")

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings_array)

    # Create index
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)

    print(f"✓ FAISS index built with {index.ntotal} vectors")

    # Save index and metadata
    index_path = config.DATA_DIR / "text_embeddings.faiss"
    metadata_path = config.DATA_DIR / "text_embeddings_metadata.pkl"

    print(f"\nSaving index to {index_path}...")
    faiss.write_index(index, str(index_path))

    print(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    print("\n" + "="*70)
    print("FAISS INDEX BUILD COMPLETE")
    print("="*70)
    print(f"\nIndex size: {index.ntotal:,} vectors")
    print(f"Dimension: {dimension}")
    print(f"Files created:")
    print(f"  - {index_path}")
    print(f"  - {metadata_path}")
    print("\nReady for fast similarity search!")
    print("="*70)

if __name__ == "__main__":
    build_faiss_index()
