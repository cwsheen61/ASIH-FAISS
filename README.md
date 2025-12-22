# Scraper Trainer

A content moderation training pipeline that scrapes content from 4chan, labels it using Gemini API, and prepares training data for local AI models.

## Purpose

This application:
1. Scrapes posts and images from 4chan/pol/
2. Encodes content locally in base64 format
3. Uses Gemini API to classify content as ALLOWED or DENIED based on hate speech criteria
4. Prepares training datasets (10,000+ training, 1,000 test samples)
5. Evaluates local models (Ollama) on the test set
6. Cleans up original data after training

## Classification Criteria

Content is classified as **DENIED** if it contains:
- Racist content or slurs
- Extreme political propaganda or calls to violence
- Homophobic content or slurs
- Transphobic content or slurs
- Other hate speech or harassment

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Gemini API key:
```bash
export GEMINI_API_KEY='your-api-key-here'
```

3. (Optional) Install Ollama for local model evaluation:
```bash
# Install Ollama from https://ollama.ai
ollama pull llava
```

## Usage

### Quick Start - Complete Workflow

Run the entire pipeline:
```bash
python scraper_trainer.py workflow --threads 20
```

### Step-by-Step Usage

1. **Scrape content from 4chan**:
```bash
python scraper_trainer.py scrape --threads 10
```

2. **Label content with Gemini**:
```bash
# Label a batch of 10
python scraper_trainer.py label

# Label all unlabeled content
python scraper_trainer.py label --all

# Custom batch size and delay
python scraper_trainer.py label --batch-size 20 --delay 0.5
```

3. **Check statistics**:
```bash
python scraper_trainer.py stats
```

4. **Prepare training data** (requires 10,000+ labeled samples):
```bash
python scraper_trainer.py prepare-training
```

5. **Evaluate model** (requires Ollama running):
```bash
python scraper_trainer.py evaluate
```

6. **Clean up original data**:
```bash
# Interactive cleanup (with confirmation)
python scraper_trainer.py cleanup

# Force cleanup without confirmation
python scraper_trainer.py cleanup --force
```

## Project Structure

```
scraper_trainer/
├── scraper_trainer.py    # Main CLI application
├── config.py             # Configuration settings
├── database.py           # SQLite database management
├── scraper.py            # 4chan scraper
├── gemini_labeler.py     # Gemini API integration
├── trainer.py            # Ollama training/evaluation
├── cleanup.py            # Data cleanup utilities
├── requirements.txt      # Python dependencies
└── data/                 # Created automatically
    ├── encoded/          # Base64 encoded images
    ├── raw/              # Raw downloads (if any)
    ├── scraper_trainer.db   # SQLite database
    └── training_data.jsonl  # Prepared training data
```

## Data Flow

1. **Scraping**: Content downloaded from 4chan → Saved as base64 encoded files
2. **Labeling**: Encoded content → Gemini API → ALLOWED/DENIED labels
3. **Training Prep**: Labeled data → Train/test split → JSONL training file
4. **Evaluation**: Test set → Ollama model → Accuracy metrics
5. **Cleanup**: Original files deleted, training data preserved

## Notes

- 4chan API has rate limits - the scraper includes delays to be respectful
- Gemini API calls cost money - monitor your usage
- Ollama models run locally and require significant disk space
- The cleanup step permanently deletes original content (but keeps the database and training file)

## Training Threshold

The system requires **10,000 labeled examples** before preparing training data, with an additional **1,000 examples** for the test set.

## Ollama Fine-tuning Note

Ollama doesn't currently support automated fine-tuning via API. The `prepare-training` command creates a JSONL file that can be used with other fine-tuning frameworks, or you can use Ollama's base models for evaluation.
