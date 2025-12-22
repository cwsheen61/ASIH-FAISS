#!/usr/bin/env python3
"""
Quick Telegram scrape for SITF testing.
"""
import asyncio
from telegram_scraper import TelegramScraper

async def main():
    scraper = TelegramScraper()
    await scraper.connect()

    # Just scrape one channel with a small limit
    channel = 'ConservativeHub'

    try:
        await scraper.scrape_channel(channel, max_messages=100)
    finally:
        await scraper.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
