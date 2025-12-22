"""Configuration settings for scraper trainer."""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ENCODED_DIR = DATA_DIR / "encoded"
DB_PATH = DATA_DIR / "scraper_trainer.db"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)
ENCODED_DIR.mkdir(exist_ok=True)

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Telegram API Configuration
TELEGRAM_API_ID = 30876956
TELEGRAM_API_HASH = "15196bdb737640ac1e69d52dc27e74bf"
TELEGRAM_PHONE = "+16034791024"

# 4chan configuration
CHAN_BOARD = "pol"
CHAN_API_BASE = "https://a.4cdn.org"
CHAN_IMAGE_BASE = "https://i.4cdn.org"

# Training configuration
TRAINING_THRESHOLD = 400  # Lowered for testing (originally 10000)
TEST_SET_SIZE = 50  # Lowered for testing (originally 1000)

# Deduplication configuration
PHASH_SIMILARITY_THRESHOLD = 5  # Hamming distance threshold (0-64, lower = more similar)
# 5 means ~92% similarity, good for catching near-duplicates

CLIP_SIMILARITY_THRESHOLD = 0.95  # Cosine similarity threshold (0-1, higher = more similar)
# 0.95 means 95% similar in semantic space, catches similar-but-cropped/edited images

# Image processing configuration
MAX_IMAGE_DIMENSION = 1024  # Resize images to max 1024px on longest side
# Gemini pricing tiers: <256px (cheap), 256-768px (medium), >768px (expensive)
# 1024px is a good balance between quality and cost
IMAGE_QUALITY = 85  # JPEG quality for compression (1-100)

# Source-specific targets (for multi-source scraping later)
SOURCE_TARGETS = {
    '4chan/pol': 4000,
    'reddit': 2000,
    'twitter': 2000,
    'telegram': 2000,
    'alt_tech': 3000  # Gab, Truth, Patriots.win combined
}

# Gemini prompt for classification
CLASSIFICATION_PROMPT = """You are a content moderation classifier for social media. Your goal is to identify memes and images that violate community guidelines.

CLASSIFY AS DENIED IF THE IMAGE CONTAINS:
- Racist content: stereotypes, slurs, or dehumanization based on race or ethnicity
- Homophobic content: anti-LGBTQ+ messaging, slurs, or mockery
- Transphobic content: anti-transgender messaging, misgendering, or mockery
- Anti-Muslim/Islamophobic content: religious bigotry, stereotypes about Muslims
- Anti-Jewish/Antisemitic content: Jewish stereotypes, Holocaust denial, conspiracy theories
- Anti-immigrant content: xenophobia, dehumanization of immigrants or refugees
- Hateful political content: calls for violence, ethnic cleansing, or promoting hate groups
- Explicit violence or gore
- Sexual content or nudity

CLASSIFY AS ALLOWED IF THE IMAGE:
- Is a regular meme without hateful content
- Contains political commentary that is not hateful or violent
- Is informative content without bigotry
- Is a clear, readable image (not blurry or corrupted)

TASK:
1. Carefully analyze the image and any text overlay or captions
2. Determine if it violates any of the DENIED criteria above
3. Classify as "DENIED" if it contains hate speech, bigotry, or explicit content
4. Classify as "ALLOWED" if it's a normal meme without violations
5. Provide a specific reason citing which guideline was violated or why it's allowed

OUTPUT FORMAT (JSON only):
{"classification": "DENIED", "reason": "Contains racist stereotypes about [group]"}
or
{"classification": "ALLOWED", "reason": "Political meme without hateful content"}"""

# Ollama configuration
OLLAMA_MODEL = "content-moderator"  # Custom trained model
OLLAMA_BASE_URL = "http://localhost:11434"
