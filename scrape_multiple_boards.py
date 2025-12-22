#!/usr/bin/env python3
"""Scrape multiple 4chan boards in sequence."""
from scraper import ChanScraper
import time

def scrape_boards(boards, threads_per_board=10):
    """Scrape multiple boards."""

    print("\n" + "="*70)
    print("MULTI-BOARD 4CHAN SCRAPING")
    print("="*70)
    print(f"Boards to scrape: {', '.join(boards)}")
    print(f"Threads per board: {threads_per_board}")
    print("="*70)

    for board in boards:
        print(f"\n\n{'='*70}")
        print(f"STARTING BOARD: /{board}/")
        print(f"{'='*70}\n")

        scraper = ChanScraper(board=board)
        scraper.scrape_catalog(max_threads=threads_per_board)

        print(f"\n✓ Completed /{board}/")
        print("Waiting 5 seconds before next board...")
        time.sleep(5)

    print("\n" + "="*70)
    print("ALL BOARDS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    # Boards known for hateful/political content
    boards_to_scrape = [
        "pol",   # Politics - racist, antisemitic, anti-immigrant content
        "b",     # Random - lots of hateful memes, no moderation
        "r9k",   # ROBOT9001 - incel, misogyny, transphobia
        "news",  # News - political discussions
        "k",     # Weapons - nationalist memes
        "his",   # History - Holocaust denial, antisemitism
        "int",   # International - racist country memes
        "biz",   # Business - anti-Soros conspiracy theories
    ]

    scrape_boards(boards_to_scrape, threads_per_board=15)
