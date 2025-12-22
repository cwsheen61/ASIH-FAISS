"""4chan scraper module."""
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


class ChanScraper:
    """Scrapes content from 4chan boards."""

    def __init__(self, board: str = config.CHAN_BOARD):
        self.board = board
        self.api_base = config.CHAN_API_BASE
        self.image_base = config.CHAN_IMAGE_BASE
        self.db = Database()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ScraperTrainer/1.0)'
        })

    def get_catalog(self) -> List[Dict]:
        """Get catalog of threads from the board."""
        url = f"{self.api_base}/{self.board}/catalog.json"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching catalog: {e}")
            return []

    def get_thread(self, thread_id: int) -> Optional[Dict]:
        """Get a specific thread with all its posts."""
        url = f"{self.api_base}/{self.board}/thread/{thread_id}.json"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching thread {thread_id}: {e}")
            return None

    def download_image(self, filename: str, thread_id: int) -> Optional[bytes]:
        """Download an image from 4chan."""
        url = f"{self.image_base}/{self.board}/{filename}"
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
            # Convert to RGB if necessary (for PNG with alpha, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            phash = imagehash.phash(image)
            return str(phash)
        except Exception as e:
            print(f"  Error calculating phash: {e}")
            return None

    def encode_and_save(self, data: bytes, post_id: str) -> Optional[str]:
        """Encode data to base64 and save to file."""
        encoded_filename = f"{post_id}.jpg.b64"  # Always save as jpg after resize
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
        if 'tim' not in post or 'ext' not in post:
            return 'text'

        ext = post['ext'].lower()

        if ext in ['.jpg', '.jpeg', '.png', '.gif']:
            return 'image'
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
                source=f"4chan/{self.board}",
                post_id=post_id,
                thread_id=str(thread_id),
                content_text=content,
                post_type=post_type,
                quarantined=True
            )
            return db_id

        # Step 3: Process images
        tim = post['tim']
        ext = post['ext']
        image_filename = f"{tim}{ext}"

        print(f" - {image_filename}", end="")

        # Download image
        image_data = self.download_image(image_filename, thread_id)
        if not image_data:
            print(f" - DOWNLOAD FAILED")
            db_id = self.db.add_post(
                source=f"4chan/{self.board}",
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
                source=f"4chan/{self.board}",
                post_id=post_id,
                thread_id=str(thread_id),
                content_text=content,
                post_type=post_type,
                quarantined=True
            )
            return db_id

        # Step 5: Calculate pHash on resized image
        print(f" - phash", end="")
        image_phash = self.calculate_phash(resized_data)
        if not image_phash:
            print(f" - PHASH FAILED")
            db_id = self.db.add_post(
                source=f"4chan/{self.board}",
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
            # Continue without CLIP if it fails

        # Step 6.5: Extract OCR text and generate text fingerprints
        print(f" - ocr", end="")
        from text_fingerprint import extract_text_from_image, generate_text_fingerprint
        ocr_text = extract_text_from_image(resized_data)
        text_hash, text_embedding = generate_text_fingerprint(content, ocr_text)

        # Step 7: Check for duplicate (pHash + CLIP)
        if self.db.is_duplicate_image(image_phash, clip_embedding):
            print(f" - DUPLICATE")
            # Still store metadata but mark as duplicate
            db_id = self.db.add_post(
                source=f"4chan/{self.board}",
                post_id=post_id,
                thread_id=str(thread_id),
                content_text=content,
                post_type=post_type,
                quarantined=True,  # Quarantine duplicates
                image_phash=image_phash,
                clip_embedding=clip_embedding
            )
            return db_id

        # Step 7: Encode and save resized image
        print(f" - storing", end="")
        encoded_path = self.encode_and_save(resized_data, f"{post_id}_{tim}")
        if not encoded_path:
            print(f" - STORAGE FAILED")
            db_id = self.db.add_post(
                source=f"4chan/{self.board}",
                post_id=post_id,
                thread_id=str(thread_id),
                content_text=content,
                post_type=post_type,
                quarantined=True
            )
            return db_id

        # Step 8: Success! Store in database
        print(f" - ✓")
        db_id = self.db.add_post(
            source=f"4chan/{self.board}",
            post_id=post_id,
            thread_id=str(thread_id),
            content_text=content,
            post_type=post_type,
            quarantined=False,
            has_image=True,
            image_filename=image_filename,
            encoded_path=encoded_path,
            image_phash=image_phash,
            clip_embedding=clip_embedding,
            ocr_text=ocr_text,
            text_hash=text_hash,
            text_embedding=text_embedding
        )

        return db_id

    def scrape_thread(self, thread_id: int) -> dict:
        """Scrape all posts from a thread. Returns statistics."""
        print(f"\nScraping thread {thread_id}...")
        thread_data = self.get_thread(thread_id)

        if not thread_data or 'posts' not in thread_data:
            print(f"  No data found for thread {thread_id}")
            return {}

        posts = thread_data['posts']
        print(f"  Found {len(posts)} posts")

        # Track this thread's stats
        thread_stats = {
            'total': 0,
            'image': 0,
            'video': 0,
            'text': 0,
            'other': 0,
            'quarantined': 0,
            'stored': 0,
            'duplicates': 0
        }

        for post in posts:
            post_type = self.classify_post_type(post)
            thread_stats['total'] += 1
            thread_stats[post_type] = thread_stats.get(post_type, 0) + 1

            # Check if it would be quarantined before processing
            if post_type != 'image':
                thread_stats['quarantined'] += 1

            self.process_post(post, thread_id)
            time.sleep(0.5)  # Be nice to the API

        return thread_stats

    def scrape_catalog(self, max_threads: int = 10):
        """Scrape threads from the catalog."""
        import time as time_module
        start_time = time_module.time()

        print(f"\n{'='*70}")
        print(f"STARTING SCRAPE: /{self.board}/")
        print(f"{'='*70}")
        print(f"Fetching catalog...")
        catalog = self.get_catalog()

        if not catalog:
            print("Failed to fetch catalog")
            return

        # Track cumulative stats
        cumulative_stats = {
            'threads': 0,
            'total': 0,
            'image': 0,
            'video': 0,
            'text': 0,
            'other': 0,
            'quarantined': 0,
            'stored': 0,
            'duplicates': 0,
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
                    # Get database stats before scraping
                    stats_before = self.db.get_stats()
                    unique_before = stats_before['unique_images']

                    # Scrape the thread
                    thread_stats = self.scrape_thread(thread_id)
                    thread_count += 1

                    # Get stats after
                    stats_after = self.db.get_stats()
                    unique_after = stats_after['unique_images']
                    new_unique = unique_after - unique_before

                    # Update cumulative
                    cumulative_stats['threads'] += 1
                    for key in ['total', 'image', 'video', 'text', 'other', 'quarantined']:
                        cumulative_stats[key] += thread_stats.get(key, 0)

                    # Print thread summary
                    print(f"  └─ Thread summary: {thread_stats.get('image', 0)} images, "
                          f"{thread_stats.get('video', 0)} videos, "
                          f"{thread_stats.get('text', 0)} text, "
                          f"{new_unique} NEW unique")

                    # Print progress after each thread
                    self._print_progress(cumulative_stats, stats_after)

                    time.sleep(1)  # Rate limiting

        # Final summary
        print(f"\n{'='*70}")
        print(f"SCRAPING COMPLETE")
        print(f"{'='*70}")
        final_stats = self.db.get_stats()
        self._print_final_summary(cumulative_stats, final_stats)

    def _print_progress(self, cumulative: dict, db_stats: dict):
        """Print progress update."""
        import time as time_module
        elapsed = time_module.time() - cumulative['start_time']
        raw_rate = cumulative['image'] / elapsed if elapsed > 0 else 0
        unique_rate = db_stats['unique_images'] / elapsed if elapsed > 0 else 0

        print(f"\n  ┌─ PROGRESS [{cumulative['threads']} threads, {elapsed:.1f}s elapsed]")
        print(f"  │ Posts processed:   {cumulative['total']}")
        print(f"  │   - Images:        {cumulative['image']} ({raw_rate:.2f}/sec)")
        print(f"  │   - Videos:        {cumulative['video']}")
        print(f"  │   - Text:          {cumulative['text']}")
        print(f"  │ Quarantined:       {cumulative['quarantined']}")
        print(f"  │ Unique images:     {db_stats['unique_images']} ({unique_rate:.2f}/sec)")
        print(f"  │ DB total:          {db_stats['total_posts']}")
        print(f"  └─")

    def _print_final_summary(self, cumulative: dict, db_stats: dict):
        """Print final summary."""
        import time as time_module
        total_time = time_module.time() - cumulative['start_time']
        raw_rate = cumulative['image'] / total_time if total_time > 0 else 0
        unique_rate = db_stats['unique_images'] / total_time if total_time > 0 else 0

        print(f"Threads scraped:     {cumulative['threads']}")
        print(f"Posts processed:     {cumulative['total']}")
        print(f"  - Images:          {cumulative['image']}")
        print(f"  - Videos:          {cumulative['video']}")
        print(f"  - Text/Other:      {cumulative['text'] + cumulative.get('other', 0)}")
        print(f"Quarantined:         {cumulative['quarantined']}")
        print(f"\nDatabase totals:")
        print(f"  Total posts:       {db_stats['total_posts']}")
        print(f"  Unique images:     {db_stats['unique_images']}")
        print(f"  Active posts:      {db_stats['active_posts']}")
        print(f"  Quarantined:       {db_stats['quarantined_posts']}")
        print(f"\nPerformance:")
        print(f"  Total time:        {total_time:.1f} seconds")
        print(f"  Raw images/sec:    {raw_rate:.2f}")
        print(f"  Unique images/sec: {unique_rate:.2f}")
        if cumulative['image'] > 0:
            dedup_rate = (1 - db_stats['unique_images'] / cumulative['image']) * 100
            print(f"  Deduplication:     {dedup_rate:.1f}%")
        print(f"{'='*70}")
