#!/usr/bin/env python3
"""Label all unlabeled images without stopping at target."""
from gemini_labeler import GeminiLabeler
import config

def main():
    labeler = GeminiLabeler()

    print("Labeling ALL unlabeled images (no target limit)...")
    print(f"API Key configured: {bool(config.GEMINI_API_KEY)}")
    print()

    # Call label_all with stop_at_target=False to label everything
    labeler.label_all(batch_size=10, delay=1.0, stop_at_target=False)

    print("\nAll images labeled!")

if __name__ == "__main__":
    main()
