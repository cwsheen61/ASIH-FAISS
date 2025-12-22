#!/usr/bin/env python3
"""
Scraper Trainer Application

A pipeline for scraping content, labeling with Gemini API,
and training local AI models for content moderation.
"""
import sys
import argparse
from scraper import ChanScraper
from gemini_labeler import GeminiLabeler
from trainer import OllamaTrainer
from cleanup import DataCleaner
from database import Database
import config


def cmd_scrape(args):
    """Scrape content from 4chan."""
    scraper = ChanScraper()
    scraper.scrape_catalog(max_threads=args.threads)


def cmd_label(args):
    """Label scraped content using Gemini API."""
    if not config.GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Please set it using: export GEMINI_API_KEY='your-api-key'")
        return

    labeler = GeminiLabeler()

    if args.all:
        labeler.label_all(batch_size=args.batch_size, delay=args.delay)
    else:
        labeler.label_batch(batch_size=args.batch_size, delay=args.delay)


def cmd_stats(args):
    """Show dataset statistics."""
    db = Database()
    stats = db.get_stats()
    labels = db.get_label_counts()
    source_breakdown = db.get_source_breakdown()

    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total posts:        {stats['total_posts']}")
    print(f"  - Active:         {stats['active_posts']}")
    print(f"  - Quarantined:    {stats['quarantined_posts']}")

    if stats.get('type_breakdown'):
        print(f"\nPost types:")
        for ptype, count in stats['type_breakdown'].items():
            print(f"  - {ptype:10} {count:5}")

    print(f"\nUnique images:      {stats['unique_images']}")
    print(f"Labeled posts:      {stats['labeled_posts']}")
    print(f"  - ALLOWED:        {stats['allowed_posts']}")
    print(f"  - DENIED:         {stats['denied_posts']}")
    print(f"Training posts:     {stats['training_posts']}")
    print(f"Test posts:         {stats['test_posts']}")

    if source_breakdown:
        print(f"\nALLOWED by source:")
        for source, count in source_breakdown.items():
            target = config.SOURCE_TARGETS.get(source, "N/A")
            if isinstance(target, int):
                pct = (count / target * 100) if target > 0 else 0
                print(f"  {source:20} {count:5} / {target:5} ({pct:5.1f}%)")
            else:
                print(f"  {source:20} {count:5}")

    print(f"\nProgress to ALLOWED target:")
    allowed_progress = stats['allowed_posts'] / config.TRAINING_THRESHOLD * 100
    print(f"  {stats['allowed_posts']}/{config.TRAINING_THRESHOLD} ({allowed_progress:.1f}%)")

    if stats['allowed_posts'] >= config.TRAINING_THRESHOLD:
        print("  ✓ Ready for training!")
    else:
        remaining = config.TRAINING_THRESHOLD - stats['allowed_posts']
        print(f"  Need {remaining} more ALLOWED examples")

    # Estimate raw posts needed
    if stats['allowed_posts'] > 0 and stats['labeled_posts'] > 0:
        allowed_rate = stats['allowed_posts'] / stats['labeled_posts']
        estimated_raw = int(remaining / allowed_rate) if allowed_rate > 0 else 0
        print(f"  Estimated raw posts needed: ~{estimated_raw}")

    print("="*60)


def cmd_prepare_training(args):
    """Prepare training data."""
    trainer = OllamaTrainer()
    trainer.prepare_training()


def cmd_evaluate(args):
    """Evaluate model on test set."""
    trainer = OllamaTrainer()
    trainer.evaluate_model()


def cmd_cleanup(args):
    """Clean up scraped data."""
    cleaner = DataCleaner()

    if args.force:
        cleaner.cleanup_all(keep_db=not args.delete_db)
    else:
        cleaner.safe_cleanup()


def cmd_workflow(args):
    """Run the complete workflow."""
    print("\n" + "="*60)
    print("SCRAPER TRAINER WORKFLOW")
    print("="*60)

    # Step 1: Scrape
    print("\n[1/5] SCRAPING CONTENT")
    print("-"*60)
    cmd_scrape(argparse.Namespace(threads=args.threads))

    # Step 2: Label
    print("\n[2/5] LABELING WITH GEMINI")
    print("-"*60)
    cmd_label(argparse.Namespace(all=True, batch_size=10, delay=1.0))

    # Step 3: Stats
    print("\n[3/5] CHECKING STATISTICS")
    print("-"*60)
    cmd_stats(argparse.Namespace())

    # Step 4: Prepare training
    db = Database()
    stats = db.get_stats()
    if stats['labeled_posts'] >= config.TRAINING_THRESHOLD:
        print("\n[4/5] PREPARING TRAINING DATA")
        print("-"*60)
        cmd_prepare_training(argparse.Namespace())

        print("\n[5/5] EVALUATING MODEL")
        print("-"*60)
        cmd_evaluate(argparse.Namespace())
    else:
        print("\n[4/5] SKIPPING TRAINING - Not enough labeled data")
        print(f"  Need {config.TRAINING_THRESHOLD - stats['labeled_posts']} more examples")

    print("\n" + "="*60)
    print("WORKFLOW COMPLETE")
    print("="*60)


def main():
    """Main entry point for the scraper trainer application."""
    parser = argparse.ArgumentParser(
        description="Scraper Trainer - Content moderation model training pipeline"
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Scrape command
    scrape_parser = subparsers.add_parser('scrape', help='Scrape content from 4chan')
    scrape_parser.add_argument(
        '--threads', type=int, default=10,
        help='Number of threads to scrape (default: 10)'
    )

    # Label command
    label_parser = subparsers.add_parser('label', help='Label content using Gemini API')
    label_parser.add_argument(
        '--all', action='store_true',
        help='Label all unlabeled content'
    )
    label_parser.add_argument(
        '--batch-size', type=int, default=10,
        help='Number of items to label per batch (default: 10)'
    )
    label_parser.add_argument(
        '--delay', type=float, default=1.0,
        help='Delay between API calls in seconds (default: 1.0)'
    )

    # Stats command
    subparsers.add_parser('stats', help='Show dataset statistics')

    # Prepare training command
    subparsers.add_parser('prepare-training', help='Prepare training data')

    # Evaluate command
    subparsers.add_parser('evaluate', help='Evaluate model on test set')

    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up scraped data')
    cleanup_parser.add_argument(
        '--force', action='store_true',
        help='Skip confirmation prompt'
    )
    cleanup_parser.add_argument(
        '--delete-db', action='store_true',
        help='Also delete the database'
    )

    # Workflow command
    workflow_parser = subparsers.add_parser('workflow', help='Run complete workflow')
    workflow_parser.add_argument(
        '--threads', type=int, default=10,
        help='Number of threads to scrape (default: 10)'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate command
    commands = {
        'scrape': cmd_scrape,
        'label': cmd_label,
        'stats': cmd_stats,
        'prepare-training': cmd_prepare_training,
        'evaluate': cmd_evaluate,
        'cleanup': cmd_cleanup,
        'workflow': cmd_workflow
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
