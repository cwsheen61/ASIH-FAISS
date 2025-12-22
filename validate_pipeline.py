#!/usr/bin/env python3
"""
Validate the fast moderation pipeline against Gemini ground truth.
Calculate false positives, false negatives, and overall accuracy.
"""
import sqlite3
import time
from fast_moderation_pipeline import FastModerationPipeline
from gemini_labeler import GeminiLabeler
import config

def main():
    print("="*70)
    print("PIPELINE VALIDATION - FALSE POSITIVES/NEGATIVES")
    print("="*70)

    pipeline = FastModerationPipeline()
    gemini = GeminiLabeler()

    # Get recent unlabeled posts (the 41 we just scraped)
    db = sqlite3.connect(config.DB_PATH)
    cursor = db.execute("""
        SELECT id, post_id, clip_embedding, content_text, encoded_path, has_image, image_phash
        FROM posts
        WHERE clip_embedding IS NOT NULL
        AND gemini_label IS NULL
        AND quarantined = 0
        ORDER BY id DESC
        LIMIT 41
    """)
    posts = cursor.fetchall()
    db.close()

    print(f"Found {len(posts)} posts to validate\n")

    results = {
        'true_positive': [],   # Correctly flagged as DENIED
        'true_negative': [],   # Correctly flagged as ALLOWED
        'false_positive': [],  # Incorrectly flagged as DENIED (should be ALLOWED)
        'false_negative': [],  # Incorrectly flagged as ALLOWED (should be DENIED)
        'uncertain_correct': [], # UNCERTAIN, should be DENIED
        'uncertain_incorrect': [], # UNCERTAIN, should be ALLOWED
    }

    for db_id, post_id, clip_emb, text, encoded_path, has_image, phash in posts:
        # Get pipeline prediction
        pipeline_result = pipeline.moderate_post(
            clip_embedding=clip_emb,
            text=text,
            image_path=encoded_path
        )
        pipeline_label = pipeline_result['final_label']

        # Get Gemini ground truth
        print(f"\nPost {post_id}:")
        print(f"  Pipeline: {pipeline_label}")

        # Check for duplicate label first
        gemini_label = None
        gemini_reason = None

        if phash:
            db = sqlite3.connect(config.DB_PATH)
            cursor = db.execute("""
                SELECT gemini_label, gemini_reason
                FROM posts
                WHERE image_phash = ? AND gemini_label IS NOT NULL
                LIMIT 1
            """, (phash,))
            existing = cursor.fetchone()
            db.close()

            if existing:
                gemini_label, gemini_reason = existing
                print(f"  Gemini: {gemini_label} (from duplicate)")

        if not gemini_label:
            image_path = encoded_path if has_image else None
            result = gemini.classify_content(text, image_path)

            if result:
                gemini_label, gemini_reason = result
                print(f"  Gemini: {gemini_label}")

                # Save to DB
                db = sqlite3.connect(config.DB_PATH)
                db.execute("""
                    UPDATE posts
                    SET gemini_label = ?, gemini_reason = ?
                    WHERE id = ?
                """, (gemini_label, gemini_reason, db_id))
                db.commit()
                db.close()
            else:
                print(f"  Gemini: FAILED - skipping")
                continue

            time.sleep(1.0)

        # Classify the result
        if pipeline_label == 'UNCERTAIN':
            if gemini_label == 'DENIED':
                results['uncertain_correct'].append({
                    'post_id': post_id,
                    'gemini_reason': gemini_reason[:100],
                    'pipeline_path': pipeline_result['decision_path']
                })
                print(f"  Result: UNCERTAIN (correctly escalated - is actually DENIED)")
            else:
                results['uncertain_incorrect'].append({
                    'post_id': post_id,
                    'gemini_reason': gemini_reason[:100],
                    'pipeline_path': pipeline_result['decision_path']
                })
                print(f"  Result: UNCERTAIN (incorrectly escalated - is actually ALLOWED)")
        elif pipeline_label == 'DENIED':
            if gemini_label == 'DENIED':
                results['true_positive'].append(post_id)
                print(f"  Result: ✓ TRUE POSITIVE (correctly flagged)")
            else:
                results['false_positive'].append({
                    'post_id': post_id,
                    'gemini_reason': gemini_reason[:100],
                    'pipeline_path': pipeline_result['decision_path'],
                    'text_flagged': pipeline_result['text_flagged'],
                    'text_reason': pipeline_result['text_reason']
                })
                print(f"  Result: ✗ FALSE POSITIVE (incorrectly flagged)")
        else:  # ALLOWED
            if gemini_label == 'ALLOWED':
                results['true_negative'].append(post_id)
                print(f"  Result: ✓ TRUE NEGATIVE (correctly allowed)")
            else:
                results['false_negative'].append({
                    'post_id': post_id,
                    'gemini_reason': gemini_reason[:100],
                    'pipeline_path': pipeline_result['decision_path']
                })
                print(f"  Result: ✗ FALSE NEGATIVE (missed hateful content)")

    # Calculate metrics
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)

    tp = len(results['true_positive'])
    tn = len(results['true_negative'])
    fp = len(results['false_positive'])
    fn = len(results['false_negative'])
    uncertain_correct = len(results['uncertain_correct'])
    uncertain_incorrect = len(results['uncertain_incorrect'])

    total = tp + tn + fp + fn + uncertain_correct + uncertain_incorrect

    print(f"\nTotal validated: {total}")
    print(f"\nDirect Predictions (ALLOWED/DENIED):")
    print(f"  True Positives:  {tp} (correctly flagged as DENIED)")
    print(f"  True Negatives:  {tn} (correctly flagged as ALLOWED)")
    print(f"  False Positives: {fp} (incorrectly flagged as DENIED)")
    print(f"  False Negatives: {fn} (missed hateful content)")

    print(f"\nUncertain Escalations:")
    print(f"  Should be DENIED:  {uncertain_correct} (good escalation)")
    print(f"  Should be ALLOWED: {uncertain_incorrect} (unnecessary escalation)")

    # Calculate accuracy treating UNCERTAIN as separate
    direct_predictions = tp + tn + fp + fn
    if direct_predictions > 0:
        accuracy_direct = (tp + tn) / direct_predictions * 100
        print(f"\nAccuracy (direct predictions only): {accuracy_direct:.1f}%")

    # Calculate if we treat uncertain_correct as "caught" and uncertain_incorrect as false positive
    caught = tp + uncertain_correct  # Total hateful content we flagged or escalated
    missed = fn  # Hateful content we allowed through
    over_flagged = fp + uncertain_incorrect  # Clean content we flagged or escalated
    correct_allowed = tn  # Clean content we correctly allowed

    print(f"\nOverall Performance:")
    print(f"  Hateful content caught:     {caught}/{caught+missed} ({caught/(caught+missed)*100 if caught+missed > 0 else 0:.1f}%)")
    print(f"  Clean content allowed:      {correct_allowed}/{correct_allowed+over_flagged} ({correct_allowed/(correct_allowed+over_flagged)*100 if correct_allowed+over_flagged > 0 else 0:.1f}%)")

    # Show examples
    if results['false_positive']:
        print(f"\n❌ FALSE POSITIVES ({len(results['false_positive'])}):")
        for item in results['false_positive'][:5]:
            print(f"\n  Post {item['post_id']}:")
            print(f"    Why flagged: {item['pipeline_path']}")
            if item['text_flagged']:
                print(f"    Text pattern: {item['text_reason']}")
            print(f"    Gemini says: {item['gemini_reason']}")

    if results['false_negative']:
        print(f"\n❌ FALSE NEGATIVES ({len(results['false_negative'])}):")
        for item in results['false_negative'][:5]:
            print(f"\n  Post {item['post_id']}:")
            print(f"    Pipeline path: {item['pipeline_path']}")
            print(f"    Gemini says: {item['gemini_reason']}")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
