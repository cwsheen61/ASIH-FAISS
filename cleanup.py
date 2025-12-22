"""Data cleanup utilities."""
import shutil
from pathlib import Path
import config
from database import Database


class DataCleaner:
    """Handles cleanup of original scraped data after training."""

    def __init__(self):
        self.db = Database()

    def cleanup_encoded_files(self):
        """Delete all encoded image files."""
        print("Cleaning up encoded files...")

        if not config.ENCODED_DIR.exists():
            print("  No encoded directory found")
            return

        files = list(config.ENCODED_DIR.glob('*'))
        count = len(files)

        for file in files:
            if file.is_file():
                file.unlink()

        print(f"  Deleted {count} encoded files")

    def cleanup_raw_files(self):
        """Delete all raw downloaded files."""
        print("Cleaning up raw files...")

        if not config.RAW_DIR.exists():
            print("  No raw directory found")
            return

        files = list(config.RAW_DIR.glob('*'))
        count = len(files)

        for file in files:
            if file.is_file():
                file.unlink()

        print(f"  Deleted {count} raw files")

    def cleanup_all(self, keep_db: bool = True):
        """Clean up all data files."""
        print("\n" + "="*50)
        print("DATA CLEANUP")
        print("="*50)

        self.cleanup_encoded_files()
        self.cleanup_raw_files()

        if not keep_db:
            if config.DB_PATH.exists():
                print("Deleting database...")
                config.DB_PATH.unlink()
                print("  Database deleted")

        print("\nCleanup complete!")
        print("="*50)

    def verify_training_complete(self) -> bool:
        """Verify that training data has been created before cleanup."""
        training_file = config.DATA_DIR / "training_data.jsonl"

        if not training_file.exists():
            print("⚠ Warning: Training data file not found")
            print("  Run 'prepare-training' before cleanup")
            return False

        stats = self.db.get_stats()
        if stats['training_posts'] == 0:
            print("⚠ Warning: No training data assigned")
            print("  Run 'prepare-training' before cleanup")
            return False

        return True

    def safe_cleanup(self):
        """Perform cleanup only if training is complete."""
        if not self.verify_training_complete():
            print("\n❌ Cleanup aborted - training not prepared")
            return

        print("\n⚠ This will delete all original scraped content")
        print("  Encoded files and images will be removed")
        print("  Database will be kept for training records")

        response = input("\nProceed with cleanup? (yes/no): ")

        if response.lower() == 'yes':
            self.cleanup_all(keep_db=True)
        else:
            print("Cleanup cancelled")
