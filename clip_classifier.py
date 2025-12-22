#!/usr/bin/env python3
"""Train a classifier on CLIP embeddings for content moderation."""
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import config
from database import Database


class CLIPClassifier:
    """Classifier that trains on CLIP embeddings."""

    def __init__(self):
        self.db = Database()
        self.model = None
        self.model_path = config.DATA_DIR / "clip_classifier.pkl"

    def load_training_data(self):
        """Load CLIP embeddings and labels from database."""
        print("Loading training data from database...")

        # Get all posts with CLIP embeddings and labels in training set
        with self.db.db_path as conn:
            import sqlite3
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.execute("""
                SELECT clip_embedding, gemini_label
                FROM posts
                WHERE clip_embedding IS NOT NULL
                AND gemini_label IS NOT NULL
                AND dataset_split = 'train'
            """)
            train_data = cursor.fetchall()

            cursor = conn.execute("""
                SELECT clip_embedding, gemini_label
                FROM posts
                WHERE clip_embedding IS NOT NULL
                AND gemini_label IS NOT NULL
                AND dataset_split = 'test'
            """)
            test_data = cursor.fetchall()
            conn.close()

        print(f"  Training examples: {len(train_data)}")
        print(f"  Test examples: {len(test_data)}")

        # Convert to numpy arrays
        X_train = np.array([json.loads(emb) for emb, _ in train_data])
        y_train = np.array([1 if label == 'DENIED' else 0 for _, label in train_data])

        X_test = np.array([json.loads(emb) for emb, _ in test_data])
        y_test = np.array([1 if label == 'DENIED' else 0 for _, label in test_data])

        return X_train, y_train, X_test, y_test

    def train(self):
        """Train logistic regression classifier on CLIP embeddings."""
        print("\n" + "="*60)
        print("TRAINING CLIP-BASED CLASSIFIER")
        print("="*60)

        # Load data
        X_train, y_train, X_test, y_test = self.load_training_data()

        if len(X_train) == 0:
            print("Error: No training data available")
            return False

        # Train logistic regression
        print("\nTraining logistic regression classifier...")
        self.model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight='balanced',  # Handle imbalanced classes
            random_state=42
        )
        self.model.fit(X_train, y_train)

        print(f"✓ Training complete")

        # Evaluate on train set
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"  Training accuracy: {train_acc:.2%}")

        # Save model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Model saved to {self.model_path}")

        return True

    def evaluate(self):
        """Evaluate the trained classifier."""
        print("\n" + "="*60)
        print("EVALUATING CLIP-BASED CLASSIFIER")
        print("="*60)

        # Load model if not already loaded
        if self.model is None:
            if not self.model_path.exists():
                print("Error: No trained model found. Run training first.")
                return False

            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("✓ Loaded trained model")

        # Load test data
        _, _, X_test, y_test = self.load_training_data()

        if len(X_test) == 0:
            print("Error: No test data available")
            return False

        print(f"\nEvaluating on {len(X_test)} test examples...")

        # Predict
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy:  {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall:    {recall:.2%}")
        print(f"F1 Score:  {f1:.2%}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives (DENIED):  {tp}")
        print(f"  False Positives:          {fp}")
        print(f"  True Negatives (ALLOWED): {tn}")
        print(f"  False Negatives:          {fn}")
        print("="*60)

        # Show some example predictions
        print("\nSample predictions:")
        for i in range(min(10, len(X_test))):
            true_label = "DENIED" if y_test[i] == 1 else "ALLOWED"
            pred_label = "DENIED" if y_pred[i] == 1 else "ALLOWED"
            confidence = y_proba[i][y_pred[i]]
            match = "✓" if y_test[i] == y_pred[i] else "✗"
            print(f"  {match} True: {true_label:7} | Pred: {pred_label:7} | Conf: {confidence:.2f}")

        return True

    def predict(self, clip_embedding_json: str) -> tuple:
        """
        Predict classification for a CLIP embedding.

        Args:
            clip_embedding_json: CLIP embedding as JSON string

        Returns:
            (label, confidence) tuple
        """
        if self.model is None:
            if not self.model_path.exists():
                raise ValueError("No trained model found")

            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)

        # Parse embedding
        embedding = np.array(json.loads(clip_embedding_json)).reshape(1, -1)

        # Predict
        pred = self.model.predict(embedding)[0]
        proba = self.model.predict_proba(embedding)[0]

        label = "DENIED" if pred == 1 else "ALLOWED"
        confidence = proba[pred]

        return label, confidence


def main():
    """Main training and evaluation function."""
    classifier = CLIPClassifier()

    # Train
    success = classifier.train()
    if not success:
        return

    # Evaluate
    classifier.evaluate()


if __name__ == "__main__":
    main()
