#!/usr/bin/env python3
"""
Database migration: Add text fingerprint columns.
Adds OCR text, text hash, and text embedding columns.
"""
import sqlite3
import config

def migrate():
    print("="*70)
    print("DATABASE MIGRATION - Adding Text Fingerprint Columns")
    print("="*70)

    db = sqlite3.connect(config.DB_PATH)
    cursor = db.cursor()

    # Check if columns already exist
    cursor.execute("PRAGMA table_info(posts)")
    columns = {col[1] for col in cursor.fetchall()}

    new_columns = []

    if 'ocr_text' not in columns:
        print("\n✓ Adding 'ocr_text' column...")
        cursor.execute("ALTER TABLE posts ADD COLUMN ocr_text TEXT")
        new_columns.append('ocr_text')

    if 'text_hash' not in columns:
        print("✓ Adding 'text_hash' column...")
        cursor.execute("ALTER TABLE posts ADD COLUMN text_hash TEXT")
        new_columns.append('text_hash')

    if 'text_embedding' not in columns:
        print("✓ Adding 'text_embedding' column...")
        cursor.execute("ALTER TABLE posts ADD COLUMN text_embedding TEXT")
        new_columns.append('text_embedding')

    if new_columns:
        # Create indices for fast lookups
        print("\n✓ Creating indices...")
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_text_hash ON posts(text_hash)")
            print("  - Index on text_hash")
        except:
            pass

        db.commit()
        print(f"\n✓ Migration complete - Added {len(new_columns)} columns")
    else:
        print("\n✓ All columns already exist - no migration needed")

    # Show updated schema
    cursor.execute("PRAGMA table_info(posts)")
    columns = cursor.fetchall()

    print("\nUpdated schema:")
    for col in columns:
        if col[1] in ['ocr_text', 'text_hash', 'text_embedding', 'image_phash', 'clip_embedding', 'gemini_label']:
            print(f"  - {col[1]}: {col[2]}")

    db.close()
    print("="*70)

if __name__ == "__main__":
    migrate()
