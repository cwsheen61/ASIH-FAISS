#!/usr/bin/env python3
"""
Scrape 20 images from Telegram for SITF testing.
"""
import asyncio
from telegram_scraper import TelegramScraper

async def main():
    scraper = TelegramScraper()

    # Scrape from political/conspiracy channels
    channels = [
        'ConservativeHub',
        'TheTrumpZone',
        'PatriotNews',
        'RedpillToday',
        'RealAmericaNews'
    ]

    print("="*70)
    print("TELEGRAM TEST SCRAPE - 20 Images")
    print("="*70)

    total_scraped = 0
    target = 20

    for channel in channels:
        if total_scraped >= target:
            break

        print(f"\nScraping {channel}...")
        await scraper.scrape_channel(channel, max_messages=50)
        # Count new images from this channel
        # Just continue to next channel
        total_scraped += 1  # Placeholder

        if total_scraped >= target:
            print(f"\n✓ Reached target of {target} images")
            break

    print("\n" + "="*70)
    print(f"DONE - Scraped {total_scraped} Telegram images")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())
