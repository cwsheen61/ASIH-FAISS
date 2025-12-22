# Content Moderation Architecture

## Problem
Current approach uses CLIP embeddings to moderate both text and images together, but:
- CLIP is designed for visual understanding, not text analysis
- Text moderation is cheaper/faster than image moderation
- Validation showed 54% accuracy - barely better than random

## Proposed Architecture: Dual-Pipeline Moderation

### Pipeline 1: Image Moderation (CLIP-based)
**For posts with images:**
- Use CLIP embeddings to generate visual features
- Train classifier on visual features only (ignore text content)
- Fast, local inference
- Handles: inappropriate images, violence, nudity, hate symbols

**Advantages:**
- CLIP excels at visual understanding
- Fast local inference (no API calls)
- Good at detecting visual hate symbols (swastikas, etc.)

### Pipeline 2: Text Moderation
**For posts with text (separate from images):**

**Option A: Simple Keyword/Regex**
- Fast, cheap, deterministic
- Match known slurs, hate speech patterns
- Good for obvious cases

**Option B: Small Text Classifier**
- Train small BERT/RoBERTa model on text labels
- Fast inference, better context understanding
- Can catch subtle hate speech

**Option C: LLM-based (Gemini API)**
- Best accuracy, understands context
- More expensive but very effective
- Use for borderline cases

### Combined Strategy

```
┌─────────────┐
│   New Post  │
└──────┬──────┘
       │
       ├─────────────────────────────────────┐
       │                                     │
       v                                     v
┌──────────────┐                    ┌──────────────┐
│  Has Image?  │                    │  Has Text?   │
└──────┬───────┘                    └──────┬───────┘
       │                                   │
       v                                   v
┌──────────────┐                    ┌──────────────┐
│ CLIP         │                    │ Text         │
│ Classifier   │                    │ Classifier   │
│              │                    │              │
│ • Visual     │                    │ • Slurs      │
│   features   │                    │ • Hate       │
│ • Hate       │                    │   patterns   │
│   symbols    │                    │ • Context    │
│ • Nudity     │                    │              │
└──────┬───────┘                    └──────┬───────┘
       │                                   │
       v                                   v
┌──────────────┐                    ┌──────────────┐
│ Image:       │                    │ Text:        │
│ ALLOWED/     │                    │ ALLOWED/     │
│ DENIED       │                    │ DENIED       │
└──────┬───────┘                    └──────┬───────┘
       │                                   │
       └───────────────┬───────────────────┘
                       v
                ┌──────────────┐
                │  Combine     │
                │  Results     │
                │              │
                │ If either    │
                │ DENIED →     │
                │ DENIED       │
                └──────┬───────┘
                       v
                ┌──────────────┐
                │ Final Label  │
                └──────────────┘
```

## Database Schema Enhancement

Already supports separation:
- `content_text` - text content
- `has_image` - boolean flag
- `clip_embedding` - visual features
- `gemini_label` - ground truth labels

**Add new fields:**
```sql
ALTER TABLE posts ADD COLUMN image_label TEXT;      -- CLIP prediction for image
ALTER TABLE posts ADD COLUMN image_confidence REAL;
ALTER TABLE posts ADD COLUMN text_label TEXT;       -- Text classifier prediction
ALTER TABLE posts ADD COLUMN text_confidence REAL;
ALTER TABLE posts ADD COLUMN combined_label TEXT;   -- Final decision
```

## Training Strategy

### Phase 1: Separate Training
1. **Image-only posts**: Train CLIP classifier on visual features
2. **Text-only posts**: Train text classifier on text content
3. **Posts with both**: Train both classifiers independently

### Phase 2: Evaluation
- Measure accuracy of each classifier on its domain
- Image classifier should be >70% accurate on visual content
- Text classifier should be >80% accurate on text content

### Phase 3: Ensemble
- Combine predictions using voting or confidence weighting
- If either classifier says DENIED with high confidence → DENIED
- Escalate borderline cases to Gemini API

## Implementation Plan

1. **Create text-only dataset** (5-10 minutes)
   - Extract posts with text but no images
   - Train simple keyword matcher or small text model

2. **Retrain CLIP classifier** (5 minutes)
   - Use only posts with images
   - Remove text content from training
   - Should improve from 54% to 65-75%

3. **Build ensemble pipeline** (15 minutes)
   - Combine text + image predictions
   - Add confidence thresholding
   - Escalation logic for borderline cases

4. **Validate on new data**
   - Test on our 35 new images
   - Measure improvement over single-pipeline approach

## Expected Results

- **Image-only posts**: 70-75% accuracy (up from 54%)
- **Text-only posts**: 80-90% accuracy (keyword matching + context)
- **Combined posts**: Best of both worlds
- **Cost**: Mostly local inference, Gemini only for edge cases
- **Speed**: Fast (CLIP + text classifier run locally)
