#!/usr/bin/env python3
"""
Test scraping with OCR and text fingerprinting enabled.
"""
from scraper import ChanScraper

def main():
    print("="*70)
    print("TESTING OCR + TEXT FINGERPRINTING IN SCRAPER")
    print("="*70)

    scraper = ChanScraper(board='pol')

    # Scrape just one thread to test
    print("\nScraping one thread from /pol/ to test OCR extraction...")

    # Get catalog
    threads = scraper.get_active_threads(limit=1)

    if not threads:
        print("No threads found")
        return

    thread_id = threads[0]
    print(f"\nScraping thread {thread_id}...")

    stats = scraper.scrape_thread(thread_id)

    print("\n" + "="*70)
    print("SCRAPE COMPLETE")
    print("="*70)
    print(f"Processed: {stats.get('total', 0)} posts")
    print(f"  Images: {stats.get('image', 0)}")
    print(f"  Quarantined: {stats.get('quarantined', 0)}")

    # Check if OCR worked
    import sqlite3
    import config

    db = sqlite3.connect(config.DB_PATH)
    cursor = db.execute("""
        SELECT COUNT(*) FROM posts
        WHERE ocr_text IS NOT NULL AND ocr_text != ''
        AND quarantined = 0
    """)
    count = cursor.fetchone()[0]
    db.close()

    print(f"\nPosts with OCR text: {count}")

    if count > 0:
        print("✓ OCR extraction working!")
    else:
        print("⚠ No OCR text extracted (images may not contain text)")

if __name__ == "__main__":
    main()
