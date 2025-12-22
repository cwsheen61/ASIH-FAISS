"""CLIP embedding generation for semantic image similarity."""
import json
from typing import Optional
from io import BytesIO
from PIL import Image
import numpy as np


# Lazy-load CLIP to avoid import errors before dependencies are installed
_clip_model = None
_clip_processor = None


def _get_clip_model():
    """Lazy-load CLIP model."""
    global _clip_model, _clip_processor

    if _clip_model is None:
        try:
            from transformers import CLIPProcessor, CLIPModel
            print("Loading CLIP model (one-time setup)...")
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model.eval()  # Set to evaluation mode
            print("CLIP model loaded successfully")
        except ImportError:
            print("Error: transformers library not installed. Run: pip install transformers torch")
            return None, None
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            return None, None

    return _clip_model, _clip_processor


def generate_clip_embedding(image_data: bytes) -> Optional[str]:
    """
    Generate CLIP embedding from image bytes.

    Args:
        image_data: Image data as bytes

    Returns:
        JSON string of embedding vector, or None if failed
    """
    model, processor = _get_clip_model()

    if model is None or processor is None:
        return None

    try:
        # Load image
        image = Image.open(BytesIO(image_data))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Process image and get embedding
        inputs = processor(images=image, return_tensors="pt")

        # Get image features (embedding)
        import torch
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        # Normalize the embedding (important for cosine similarity)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Convert to list and return as JSON
        embedding = image_features.squeeze().cpu().numpy().tolist()
        return json.dumps(embedding)

    except Exception as e:
        print(f"  Error generating CLIP embedding: {e}")
        return None


def clip_similarity(embedding1_json: str, embedding2_json: str) -> float:
    """
    Calculate cosine similarity between two CLIP embeddings.

    Args:
        embedding1_json: First embedding as JSON string
        embedding2_json: Second embedding as JSON string

    Returns:
        Similarity score from 0 to 1 (1 = identical)
    """
    try:
        vec1 = np.array(json.loads(embedding1_json))
        vec2 = np.array(json.loads(embedding2_json))

        # Cosine similarity
        return float(np.dot(vec1, vec2))  # Already normalized, so just dot product

    except (ValueError, TypeError, json.JSONDecodeError):
        return 0.0
