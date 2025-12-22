#!/bin/bash
# Monitor labeling progress

while true; do
  clear
  echo "========================================"
  echo "LABELING PROGRESS - $(date '+%H:%M:%S')"
  echo "========================================"
  echo ""

  # Get database stats
  python3 << 'EOF'
import sqlite3
import config

db = sqlite3.connect(config.DB_PATH)
cursor = db.cursor()

# Total posts
cursor.execute("SELECT COUNT(*) FROM posts WHERE quarantined = 0")
total = cursor.fetchone()[0]

# Labeled posts
cursor.execute("SELECT COUNT(*) FROM posts WHERE quarantined = 0 AND gemini_label IS NOT NULL")
labeled = cursor.fetchone()[0]

# ALLOWED and DENIED
cursor.execute("SELECT COUNT(*) FROM posts WHERE quarantined = 0 AND gemini_label = 'ALLOWED'")
allowed = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM posts WHERE quarantined = 0 AND gemini_label = 'DENIED'")
denied = cursor.fetchone()[0]

unlabeled = total - labeled
percent = (labeled / total * 100) if total > 0 else 0

print(f"Total active posts:  {total}")
print(f"Labeled:             {labeled} ({percent:.1f}%)")
print(f"  - ALLOWED:         {allowed}")
print(f"  - DENIED:          {denied}")
print(f"Unlabeled:           {unlabeled}")
print(f"")
print(f"Target: {config.TRAINING_THRESHOLD + config.TEST_SET_SIZE} ALLOWED posts")
print(f"Progress: {allowed}/{config.TRAINING_THRESHOLD + config.TEST_SET_SIZE} ({allowed/(config.TRAINING_THRESHOLD + config.TEST_SET_SIZE)*100:.1f}%)")

db.close()
EOF

  echo ""
  echo "========================================"
  sleep 30
done
