"""8kun imageboard scraper module."""
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional
import base64
from io import BytesIO
from PIL import Image
import imagehash
import config
from database import Database
from clip_encoder import generate_clip_embedding


class KunScraper:
    """Scrapes content from 8kun boards."""

    def __init__(self, board: str = "pnd"):  # /pnd/ - Politics, News, Debate
        self.board = board
        self.api_base = "https://8kun.top"
        self.db = Database()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ScraperTrainer/1.0)'
        })

    def get_catalog(self) -> List[Dict]:
        """Get catalog of threads from the board."""
        url = f"{self.api_base}/{self.board}/catalog.json"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching catalog: {e}")
            return []

    def get_thread(self, thread_id: int) -> Optional[Dict]:
        """Get a specific thread with all its posts."""
        url = f"{self.api_base}/{self.board}/res/{thread_id}.json"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching thread {thread_id}: {e}")
            return None

    def download_image(self, filename: str, thread_id: int = None) -> Optional[bytes]:
        """Download an image from 8kun."""
        # 8kun uses /file_store/ path for media
        url = f"{self.api_base}/file_store/{filename}"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            print(f"Error downloading image {filename}: {e}")
            return None

    def resize_image(self, image_data: bytes) -> Optional[bytes]:
        """Resize image to standard resolution for storage and processing."""
        try:
            image = Image.open(BytesIO(image_data))

            # Convert to RGB if necessary
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')

            # Resize if needed
            max_dim = max(image.size)
            if max_dim > config.MAX_IMAGE_DIMENSION:
                ratio = config.MAX_IMAGE_DIMENSION / max_dim
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Save to bytes with compression
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

    def classify_post_type(self, post: Dict) -> str:
        """Classify post content type."""
        # 8kun uses different field names than 4chan
        if 'filename' not in post or not post.get('filename'):
            return 'text'

        ext = post.get('ext', '').lower()

        if ext in ['.jpg', '.jpeg', '.png']:
            return 'image'
        elif ext in ['.gif']:
            return 'gif'
        elif ext in ['.webm', '.mp4']:
            return 'video'
        else:
            return 'other'

    def process_post(self, post: Dict, thread_id: int) -> int:
        """Process a single post and save to database."""
        post_id = str(post.get('no', ''))
        content = post.get('com', '')

        # Step 1: Classify post type
        post_type = self.classify_post_type(post)
        print(f"  Post {post_id} [{post_type}]", end="")

        # Step 2: Quarantine non-images
        if post_type != 'image':
            print(f" - QUARANTINED")
            db_id = self.db.add_post(
                source=f"8kun/{self.board}",
                post_id=post_id,
                thread_id=str(thread_id),
                content_text=content,
                post_type=post_type,
                quarantined=True
            )
            return db_id

        # Step 3: Process images
        filename = post.get('filename', '')
        ext = post.get('ext', '')
        tim = post.get('tim', post_id)  # 8kun might use different field
        image_filename = f"{tim}{ext}"

        print(f" - {image_filename}", end="")

        # Download image
        image_data = self.download_image(image_filename, thread_id)
        if not image_data:
            print(f" - DOWNLOAD FAILED")
            db_id = self.db.add_post(
                source=f"8kun/{self.board}",
                post_id=post_id,
                thread_id=str(thread_id),
                content_text=content,
                post_type=post_type,
                quarantined=True
            )
            return db_id

        # Step 4: Resize immediately
        print(f" - resizing", end="")
        resized_data = self.resize_image(image_data)
        if not resized_data:
            print(f" - RESIZE FAILED")
            db_id = self.db.add_post(
                source=f"8kun/{self.board}",
                post_id=post_id,
                thread_id=str(thread_id),
                content_text=content,
                post_type=post_type,
                quarantined=True
            )
            return db_id

        # Step 5: Calculate pHash
        print(f" - phash", end="")
        image_phash = self.calculate_phash(resized_data)
        if not image_phash:
            print(f" - PHASH FAILED")
            db_id = self.db.add_post(
                source=f"8kun/{self.board}",
                post_id=post_id,
                thread_id=str(thread_id),
                content_text=content,
                post_type=post_type,
                quarantined=True
            )
            return db_id

        # Step 6: Generate CLIP embedding
        print(f" - clip", end="")
        clip_embedding = generate_clip_embedding(resized_data)
        if not clip_embedding:
            print(f" - CLIP FAILED (continuing anyway)")

        # Step 7: Check for duplicate
        if self.db.is_duplicate_image(image_phash, clip_embedding):
            print(f" - DUPLICATE")
            db_id = self.db.add_post(
                source=f"8kun/{self.board}",
                post_id=post_id,
                thread_id=str(thread_id),
                content_text=content,
                post_type=post_type,
                quarantined=True,
                image_phash=image_phash,
                clip_embedding=clip_embedding
            )
            return db_id

        # Step 8: Encode and save
        print(f" - storing", end="")
        encoded_path = self.encode_and_save(resized_data, f"{post_id}_{tim}")
        if not encoded_path:
            print(f" - STORAGE FAILED")
            db_id = self.db.add_post(
                source=f"8kun/{self.board}",
                post_id=post_id,
                thread_id=str(thread_id),
                content_text=content,
                post_type=post_type,
                quarantined=True
            )
            return db_id

        # Success!
        print(f" - ✓")
        db_id = self.db.add_post(
            source=f"8kun/{self.board}",
            post_id=post_id,
            thread_id=str(thread_id),
            content_text=content,
            post_type=post_type,
            quarantined=False,
            has_image=True,
            image_filename=image_filename,
            encoded_path=encoded_path,
            image_phash=image_phash,
            clip_embedding=clip_embedding
        )

        return db_id

    def scrape_thread(self, thread_id: int) -> dict:
        """Scrape all posts from a thread."""
        print(f"\nScraping thread {thread_id}...")
        thread_data = self.get_thread(thread_id)

        if not thread_data or 'posts' not in thread_data:
            print(f"  No data found for thread {thread_id}")
            return {}

        posts = thread_data['posts']
        print(f"  Found {len(posts)} posts")

        thread_stats = {
            'total': 0,
            'image': 0,
            'video': 0,
            'text': 0,
            'other': 0,
            'quarantined': 0,
            'stored': 0
        }

        for post in posts:
            post_type = self.classify_post_type(post)
            thread_stats['total'] += 1
            thread_stats[post_type] = thread_stats.get(post_type, 0) + 1

            if post_type != 'image':
                thread_stats['quarantined'] += 1

            self.process_post(post, thread_id)
            time.sleep(0.5)

        return thread_stats

    def scrape_catalog(self, max_threads: int = 10):
        """Scrape threads from the catalog."""
        import time as time_module
        start_time = time_module.time()

        print(f"\n{'='*70}")
        print(f"STARTING SCRAPE: 8kun /{self.board}/")
        print(f"{'='*70}")
        print(f"Fetching catalog...")
        catalog = self.get_catalog()

        if not catalog:
            print("Failed to fetch catalog")
            return

        cumulative_stats = {
            'threads': 0,
            'total': 0,
            'image': 0,
            'video': 0,
            'text': 0,
            'other': 0,
            'quarantined': 0,
            'stored': 0,
            'start_time': start_time
        }

        thread_count = 0
        for page in catalog:
            if thread_count >= max_threads:
                break

            threads = page.get('threads', [])
            for thread in threads:
                if thread_count >= max_threads:
                    break

                thread_id = thread.get('no')
                if thread_id:
                    stats_before = self.db.get_stats()
                    unique_before = stats_before['unique_images']

                    thread_stats = self.scrape_thread(thread_id)
                    thread_count += 1

                    stats_after = self.db.get_stats()
                    unique_after = stats_after['unique_images']
                    new_unique = unique_after - unique_before

                    for key in ['total', 'image', 'video', 'text', 'other', 'quarantined']:
                        cumulative_stats[key] += thread_stats.get(key, 0)
                    cumulative_stats['threads'] += 1

                    print(f"  └─ Thread summary: {thread_stats.get('image', 0)} images, "
                          f"{thread_stats.get('video', 0)} videos, "
                          f"{thread_stats.get('text', 0)} text, "
                          f"{new_unique} NEW unique")

                    time.sleep(1)

        print(f"\n{'='*70}")
        print(f"SCRAPING COMPLETE")
        print(f"{'='*70}")
        final_stats = self.db.get_stats()
        elapsed = time_module.time() - start_time
        print(f"Threads scraped:     {cumulative_stats['threads']}")
        print(f"Posts processed:     {cumulative_stats['total']}")
        print(f"  - Images:          {cumulative_stats['image']}")
        print(f"  - Videos:          {cumulative_stats['video']}")
        print(f"Database totals:")
        print(f"  Total posts:       {final_stats['total_posts']}")
        print(f"  Unique images:     {final_stats['unique_images']}")
        print(f"Time elapsed:        {elapsed:.1f}s")
        print(f"{'='*70}")


if __name__ == "__main__":
    scraper = KunScraper()
    scraper.scrape_catalog(max_threads=5)
