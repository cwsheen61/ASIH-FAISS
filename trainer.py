"""Ollama model training and evaluation."""
import json
import base64
import requests
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import config
from database import Database


class OllamaTrainer:
    """Handles training and evaluation using Ollama."""

    def __init__(self):
        self.db = Database()
        self.base_url = config.OLLAMA_BASE_URL
        self.model = config.OLLAMA_MODEL

    def check_ollama(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def decode_image(self, encoded_path: str) -> Optional[str]:
        """Decode base64 image and return as base64 string for Ollama."""
        try:
            with open(encoded_path, 'rb') as f:
                encoded_data = f.read()
            # Already base64 encoded, just decode and re-encode to ensure format
            decoded = base64.b64decode(encoded_data)
            return base64.b64encode(decoded).decode('utf-8')
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None

    def create_training_data_file(self) -> Path:
        """Create a JSONL file with training data for fine-tuning."""
        training_data = self.db.get_training_data()
        output_file = config.DATA_DIR / "training_data.jsonl"

        print(f"Creating training data file with {len(training_data)} examples...")

        with open(output_file, 'w') as f:
            for encoded_path, label, text, has_image in training_data:
                prompt = "Classify this content as ALLOWED or DENIED based on harmful content criteria."

                if text:
                    prompt += f"\n\nText: {text}"

                # Create training example
                example = {
                    "prompt": prompt,
                    "response": label,
                    "has_image": has_image,
                    "image_path": encoded_path if has_image else None
                }

                f.write(json.dumps(example) + '\n')

        print(f"Training data saved to {output_file}")
        return output_file

    def classify_with_ollama(self, text: Optional[str], encoded_image_path: Optional[str]) -> Tuple[Optional[str], float]:
        """Classify content using Ollama model."""
        prompt = config.CLASSIFICATION_PROMPT

        if text:
            prompt += f"\n\nText content: {text}"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        # Add image if present
        if encoded_image_path:
            image_b64 = self.decode_image(encoded_image_path)
            if image_b64:
                payload["images"] = [image_b64]

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            response_text = result.get('response', '').strip().upper()

            # Parse response
            if 'DENIED' in response_text:
                return 'DENIED', 1.0
            elif 'ALLOWED' in response_text:
                return 'ALLOWED', 1.0
            else:
                return None, 0.0

        except requests.RequestException as e:
            print(f"Error calling Ollama: {e}")
            return None, 0.0

    def evaluate_model(self) -> Dict:
        """Evaluate the model on test set."""
        if not self.check_ollama():
            print("Error: Ollama is not running")
            print(f"Please start Ollama and ensure {self.model} is available")
            return {}

        test_data = self.db.get_test_data()

        if not test_data:
            print("No test data available")
            return {}

        print(f"\nEvaluating model on {len(test_data)} test examples...")

        correct = 0
        total = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for i, (encoded_path, true_label, text, has_image) in enumerate(test_data):
            image_path = encoded_path if has_image else None

            print(f"  [{i+1}/{len(test_data)}] ", end="")
            predicted_label, confidence = self.classify_with_ollama(text, image_path)

            if predicted_label:
                total += 1
                if predicted_label == true_label:
                    correct += 1
                    print(f"✓ {predicted_label}")
                else:
                    print(f"✗ Predicted: {predicted_label}, Actual: {true_label}")

                # Calculate confusion matrix
                if predicted_label == 'DENIED' and true_label == 'DENIED':
                    true_positives += 1
                elif predicted_label == 'DENIED' and true_label == 'ALLOWED':
                    false_positives += 1
                elif predicted_label == 'ALLOWED' and true_label == 'ALLOWED':
                    true_negatives += 1
                elif predicted_label == 'ALLOWED' and true_label == 'DENIED':
                    false_negatives += 1
            else:
                print("✗ Failed to classify")

        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives
        }

        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall:    {recall:.2%}")
        print(f"F1 Score:  {f1:.2%}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {true_positives}")
        print(f"  False Positives: {false_positives}")
        print(f"  True Negatives:  {true_negatives}")
        print(f"  False Negatives: {false_negatives}")
        print("="*50)

        return results

    def prepare_training(self):
        """Prepare data for training."""
        stats = self.db.get_stats()

        print("\nDataset Statistics:")
        print(f"  Total posts: {stats['total_posts']}")
        print(f"  Labeled posts: {stats['labeled_posts']}")

        if stats['labeled_posts'] < config.TRAINING_THRESHOLD:
            print(f"\n⚠ Warning: Need at least {config.TRAINING_THRESHOLD} labeled examples")
            print(f"  Current: {stats['labeled_posts']}")
            print(f"  Remaining: {config.TRAINING_THRESHOLD - stats['labeled_posts']}")
            return False

        # Assign train/test splits
        print("\nAssigning train/test splits...")
        self.db.assign_dataset_splits()

        updated_stats = self.db.get_stats()
        print(f"  Training set: {updated_stats['training_posts']}")
        print(f"  Test set: {updated_stats['test_posts']}")

        # Create training file
        training_file = self.create_training_data_file()

        print("\n" + "="*50)
        print("TRAINING PREPARATION COMPLETE")
        print("="*50)
        print(f"Training data file: {training_file}")
        print("\nNote: Ollama doesn't support automated fine-tuning via API.")
        print("The training data has been prepared in JSONL format.")
        print("\nTo train a custom model, you'll need to:")
        print("1. Use the prepared data with a fine-tuning framework")
        print("2. Or use Ollama with a pre-trained model for inference")
        print("\nFor now, you can evaluate using the base model.")

        return True
