"""Telegram channel scraper module."""
import asyncio
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Optional
from PIL import Image
import imagehash
from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
import config
from database import Database
from clip_encoder import generate_clip_embedding


class TelegramScraper:
    """Scrapes content from Telegram channels."""

    def __init__(self):
        self.api_id = config.TELEGRAM_API_ID
        self.api_hash = config.TELEGRAM_API_HASH
        self.db = Database()
        self.session_file = config.DATA_DIR / "telegram_session"
        self.client = None

    async def connect(self):
        """Connect to Telegram and authenticate."""
        self.client = TelegramClient(
            str(self.session_file),
            self.api_id,
            self.api_hash
        )

        phone = config.TELEGRAM_PHONE
        await self.client.start(phone=phone)
        print("✓ Connected to Telegram")

    async def disconnect(self):
        """Disconnect from Telegram."""
        if self.client:
            await self.client.disconnect()

    def resize_image(self, image_data: bytes) -> Optional[bytes]:
        """Resize image to standard resolution."""
        try:
            image = Image.open(BytesIO(image_data))

            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')

            max_dim = max(image.size)
            if max_dim > config.MAX_IMAGE_DIMENSION:
                ratio = config.MAX_IMAGE_DIMENSION / max_dim
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            output = BytesIO()
            if image.mode == 'L':
                image.save(output, format='PNG', optimize=True)
            else:
                image.save(output, format='JPEG', quality=config.IMAGE_QUALITY, optimize=True)

            return output.getvalue()
        except Exception as e:
            print(f"  Error resizing image: {e}")
            return None

    def calculate_phash(self, image_data: bytes) -> Optional[str]:
        """Calculate perceptual hash of an image."""
        try:
            image = Image.open(BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            phash = imagehash.phash(image)
            return str(phash)
        except Exception as e:
            print(f"  Error calculating phash: {e}")
            return None

    def encode_and_save(self, data: bytes, post_id: str) -> Optional[str]:
        """Encode data to base64 and save to file."""
        encoded_filename = f"{post_id}.jpg.b64"
        encoded_path = config.ENCODED_DIR / encoded_filename

        try:
            encoded_data = base64.b64encode(data)
            with open(encoded_path, 'wb') as f:
                f.write(encoded_data)
            return str(encoded_path)
        except Exception as e:
            print(f"  Error encoding/saving data: {e}")
            return None

    def classify_media_type(self, media) -> str:
        """Classify media type."""
        if isinstance(media, MessageMediaPhoto):
            return 'image'
        elif isinstance(media, MessageMediaDocument):
            if media.document.mime_type.startswith('image/'):
                return 'image'
            elif media.document.mime_type.startswith('video/'):
                return 'video'
            elif media.document.mime_type == 'image/gif':
                return 'gif'
        return 'other'

    async def process_message(self, message, channel_username: str) -> int:
        """Process a single Telegram message."""
        message_id = message.id
        text = message.text or message.message or ""

        # Check if message has media
        if not message.media:
            print(f"  Message {message_id} [text] - QUARANTINED")
            db_id = self.db.add_post(
                source=f"telegram/{channel_username}",
                post_id=str(message_id),
                thread_id="0",
                content_text=text,
                post_type='text',
                quarantined=True
            )
            return db_id

        # Classify media type
        media_type = self.classify_media_type(message.media)
        print(f"  Message {message_id} [{media_type}]", end="")

        # Only process images
        if media_type != 'image':
            print(f" - QUARANTINED")
            db_id = self.db.add_post(
                source=f"telegram/{channel_username}",
                post_id=str(message_id),
                thread_id="0",
                content_text=text,
                post_type=media_type,
                quarantined=True
            )
            return db_id

        # Download image
        print(f" - downloading", end="")
        try:
            image_data = await message.download_media(file=bytes)
            if not image_data:
                print(f" - DOWNLOAD FAILED")
                db_id = self.db.add_post(
                    source=f"telegram/{channel_username}",
                    post_id=str(message_id),
                    thread_id="0",
                    content_text=text,
                    post_type=media_type,
                    quarantined=True
                )
                return db_id
        except Exception as e:
            print(f" - DOWNLOAD FAILED: {e}")
            db_id = self.db.add_post(
                source=f"telegram/{channel_username}",
                post_id=str(message_id),
                thread_id="0",
                content_text=text,
                post_type=media_type,
                quarantined=True
            )
            return db_id

        # Resize
        print(f" - resizing", end="")
        resized_data = self.resize_image(image_data)
        if not resized_data:
            print(f" - RESIZE FAILED")
            db_id = self.db.add_post(
                source=f"telegram/{channel_username}",
                post_id=str(message_id),
                thread_id="0",
                content_text=text,
                post_type=media_type,
                quarantined=True
            )
            return db_id

        # Calculate pHash
        print(f" - phash", end="")
        image_phash = self.calculate_phash(resized_data)
        if not image_phash:
            print(f" - PHASH FAILED")
            db_id = self.db.add_post(
                source=f"telegram/{channel_username}",
                post_id=str(message_id),
                thread_id="0",
                content_text=text,
                post_type=media_type,
                quarantined=True
            )
            return db_id

        # Generate CLIP embedding
        print(f" - clip", end="")
        clip_embedding = generate_clip_embedding(resized_data)
        if not clip_embedding:
            print(f" - CLIP FAILED (continuing anyway)")

        # Check for duplicate
        if self.db.is_duplicate_image(image_phash, clip_embedding):
            print(f" - DUPLICATE")
            db_id = self.db.add_post(
                source=f"telegram/{channel_username}",
                post_id=str(message_id),
                thread_id="0",
                content_text=text,
                post_type=media_type,
                quarantined=True,
                image_phash=image_phash,
                clip_embedding=clip_embedding
            )
            return db_id

        # Encode and save
        print(f" - storing", end="")
        encoded_path = self.encode_and_save(resized_data, f"tg_{channel_username}_{message_id}")
        if not encoded_path:
            print(f" - STORAGE FAILED")
            db_id = self.db.add_post(
                source=f"telegram/{channel_username}",
                post_id=str(message_id),
                thread_id="0",
                content_text=text,
                post_type=media_type,
                quarantined=True
            )
            return db_id

        # Success!
        print(f" - ✓")
        db_id = self.db.add_post(
            source=f"telegram/{channel_username}",
            post_id=str(message_id),
            thread_id="0",
            content_text=text,
            post_type=media_type,
            quarantined=False,
            has_image=True,
            image_filename=f"tg_{message_id}",
            encoded_path=encoded_path,
            image_phash=image_phash,
            clip_embedding=clip_embedding
        )

        return db_id

    async def scrape_channel(self, channel_username: str, max_messages: int = 100):
        """Scrape messages from a Telegram channel."""
        import time as time_module
        start_time = time_module.time()

        print(f"\n{'='*70}")
        print(f"SCRAPING TELEGRAM CHANNEL: @{channel_username}")
        print(f"{'='*70}")

        try:
            # Get channel entity
            entity = await self.client.get_entity(channel_username)
            print(f"Channel: {entity.title}")
            print(f"Fetching up to {max_messages} messages...")

            stats_before = self.db.get_stats()
            unique_before = stats_before['unique_images']

            # Track stats
            channel_stats = {
                'total': 0,
                'image': 0,
                'video': 0,
                'text': 0,
                'other': 0,
                'stored': 0
            }

            # Iterate through messages
            async for message in self.client.iter_messages(entity, limit=max_messages):
                if message:
                    # Classify for stats
                    if message.media:
                        media_type = self.classify_media_type(message.media)
                    else:
                        media_type = 'text'

                    channel_stats['total'] += 1
                    channel_stats[media_type] = channel_stats.get(media_type, 0) + 1

                    await self.process_message(message, channel_username)
                    await asyncio.sleep(0.5)  # Rate limiting

            # Print summary
            stats_after = self.db.get_stats()
            unique_after = stats_after['unique_images']
            new_unique = unique_after - unique_before

            print(f"\n{'='*70}")
            print(f"SCRAPING COMPLETE")
            print(f"{'='*70}")
            print(f"Messages processed:  {channel_stats['total']}")
            print(f"  - Images:          {channel_stats.get('image', 0)}")
            print(f"  - Videos:          {channel_stats.get('video', 0)}")
            print(f"  - Text:            {channel_stats.get('text', 0)}")
            print(f"New unique images:   {new_unique}")
            print(f"Time elapsed:        {time_module.time() - start_time:.1f}s")
            print(f"{'='*70}")

        except Exception as e:
            print(f"Error scraping channel: {e}")

    async def scrape_channels(self, channels: List[str], max_messages_per_channel: int = 100):
        """Scrape multiple Telegram channels."""
        await self.connect()

        try:
            for channel in channels:
                await self.scrape_channel(channel, max_messages_per_channel)
        finally:
            await self.disconnect()


async def main():
    """Main function to test scraper."""
    scraper = TelegramScraper()

    # Political/conspiracy/hate content channels
    channels = [
        "politicalmemes",      # Already scraped, but get new content
        "patriotmemes",        # Nationalist/MAGA content
        "conspiracymemes",     # Conspiracy theories
        "stolenelection",      # Election denial
        "buildthewall",        # Anti-immigrant
        "americafirst",        # Nick Fuentes - far-right
        "stoptheinvasion",     # Anti-immigrant
        "whitehouse",          # Political
        "worldpolitics",       # International politics
        "liberalnews",         # Political (may have anti-liberal memes)
    ]

    await scraper.scrape_channels(channels, max_messages_per_channel=200)


if __name__ == "__main__":
    asyncio.run(main())
