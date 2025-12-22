#!/usr/bin/env python3
"""Create and train a custom Ollama model from training data."""
import json
import subprocess
from pathlib import Path
import config
from database import Database


def create_modelfile():
    """Create a Modelfile for training with examples."""
    db = Database()
    training_data = db.get_training_data()

    # Limit examples to avoid huge modelfile (use first 100)
    sample_size = min(100, len(training_data))
    training_sample = training_data[:sample_size]

    print(f"Creating Modelfile with {sample_size} training examples...")

    # Build system prompt with examples (all on single line for Modelfile)
    system_prompt = "You are a content moderation classifier. Classify images and text as ALLOWED or DENIED. "
    system_prompt += "ALLOWED: Clear, high-quality memes or informative posts. "
    system_prompt += "DENIED: Blurry, corrupted, UI elements, or unsafe content. "
    system_prompt += "Respond in JSON: {\"classification\": \"ALLOWED/DENIED\", \"reason\": \"explanation\"}. "
    system_prompt += "Training examples: "

    # Add training examples to system prompt
    for i, (encoded_path, label, text, has_image) in enumerate(training_sample):
        if i >= sample_size:
            break

        if text and len(text) > 0:
            # Clean text: remove HTML, newlines, quotes
            clean_text = text[:80].replace('<', '').replace('>', '').replace('"', "'").replace('\n', ' ')
            system_prompt += f"Ex{i+1}: '{clean_text}' => {label}. "
        else:
            system_prompt += f"Ex{i+1}: [image] => {label}. "

    # Build the modelfile
    modelfile_content = f"""FROM {config.OLLAMA_MODEL}

SYSTEM {system_prompt}

PARAMETER temperature 0.3
PARAMETER top_p 0.9
"""

    modelfile_path = config.DATA_DIR / "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)

    print(f"Modelfile created at {modelfile_path}")
    return modelfile_path


def train_model(modelfile_path: Path, model_name: str = "content-moderator"):
    """Create a custom model using ollama create."""
    print(f"\nCreating custom model '{model_name}'...")
    print("This may take a few minutes...")

    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        print(f"\n✓ Successfully created model: {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating model: {e}")
        print(e.stderr)
        return False


def main():
    """Main training function."""
    print("="*60)
    print("TRAINING CUSTOM CONTENT MODERATION MODEL")
    print("="*60)

    # Create modelfile with training examples
    modelfile_path = create_modelfile()

    # Create custom model
    custom_model_name = "content-moderator"
    success = train_model(modelfile_path, custom_model_name)

    if success:
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Your custom model '{custom_model_name}' is ready.")
        print(f"\nTo use it, update config.py:")
        print(f'  OLLAMA_MODEL = "{custom_model_name}"')
        print(f"\nThen run: python scraper_trainer.py evaluate")
    else:
        print("\n✗ Training failed")
        return False

    return True


if __name__ == "__main__":
    main()
