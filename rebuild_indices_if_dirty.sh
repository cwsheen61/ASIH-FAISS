#!/bin/bash
#
# Automated FAISS Index Rebuild Script
#
# This script checks if indices are marked as dirty (new content added)
# and rebuilds them if needed, then restarts the API to load new indices.
#
# Usage:
#   ./rebuild_indices_if_dirty.sh
#
# Cron example (runs every hour):
#   0 * * * * /home/cwsheen/Workspace/scraper_trainer/rebuild_indices_if_dirty.sh >> /tmp/index_rebuild.log 2>&1
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRTY_FLAG="$SCRIPT_DIR/data/.indices_dirty"
LOG_FILE="/tmp/index_rebuild.log"
SOCKET_PATH="/tmp/moderator.sock"

cd "$SCRIPT_DIR"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if indices are dirty
if [ ! -f "$DIRTY_FLAG" ]; then
    log "Indices are clean, no rebuild needed"
    exit 0
fi

log "==================================================================="
log "INDICES MARKED AS DIRTY - STARTING REBUILD"
log "==================================================================="

# Count how many new entries since last rebuild
NEW_ENTRIES=$(sqlite3 data/scraper_trainer.db "SELECT COUNT(*) FROM posts WHERE source LIKE 'gocial/%'")
log "New production entries: $NEW_ENTRIES"

# Rebuild text index
log "Rebuilding text FAISS index..."
if python build_faiss_index.py >> "$LOG_FILE" 2>&1; then
    log "✓ Text index rebuilt successfully"
else
    log "✗ Text index rebuild failed"
    exit 1
fi

# Rebuild image index
log "Rebuilding image FAISS index..."
if python build_image_faiss_index.py >> "$LOG_FILE" 2>&1; then
    log "✓ Image index rebuilt successfully"
else
    log "✗ Image index rebuild failed"
    exit 1
fi

# Restart API to load new indices
log "Restarting moderator API..."

# Kill existing process
if pkill -f "moderator_api.py"; then
    log "✓ Stopped old API process"
    sleep 2
else
    log "⚠ No existing API process found"
fi

# Remove old socket
if [ -S "$SOCKET_PATH" ]; then
    rm "$SOCKET_PATH"
    log "✓ Removed old socket"
fi

# Start new process
nohup python moderator_api.py --socket "$SOCKET_PATH" > /tmp/moderator_api.log 2>&1 &
NEW_PID=$!
log "✓ Started new API process (PID: $NEW_PID)"

# Wait for API to be ready
sleep 5

# Health check
if curl --unix-socket "$SOCKET_PATH" http://localhost/health 2>/dev/null | grep -q "healthy"; then
    log "✓ API health check passed"
else
    log "✗ API health check failed"
    exit 1
fi

# Remove dirty flag
rm "$DIRTY_FLAG"
log "✓ Cleared dirty flag"

log "==================================================================="
log "REBUILD COMPLETE - API RUNNING WITH UPDATED INDICES"
log "==================================================================="

# Show stats
STATS=$(curl --unix-socket "$SOCKET_PATH" http://localhost/health 2>/dev/null | python -m json.tool)
log "Current index sizes:"
log "$STATS"

exit 0
