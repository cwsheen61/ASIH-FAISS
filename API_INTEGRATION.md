# Content Moderation API Integration

Fast Unix socket API for Gocial integration with the fingerprint-based content moderation system.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ADMIN APPLICATION                        │
│  (Background - Separate from Gocial)                        │
│                                                              │
│  • Scrape hate sources (4chan, etc.)                        │
│  • Generate fingerprints (pHash, CLIP, embeddings)          │
│  • Moderate with Gemini/Ollama                              │
│  • Build/update FAISS indices                               │
│                                                              │
│  Files: scraper.py, gemini_labeler.py,                      │
│         moderate_text_posts.py, build_faiss_index.py        │
└──────────────────────┬──────────────────────────────────────┘
                       │ Writes to
                       ▼
              ┌────────────────┐
              │   DATABASE     │
              │   + FAISS      │
              │   (Read-Only)  │
              └────────┬───────┘
                       │ Reads from
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              MODERATION API SERVICE                         │
│  (Unix Socket Server)                                        │
│                                                              │
│  • Fast FAISS lookups (~5ms)                                │
│  • REST API over Unix socket (<1ms overhead)                │
│  • Stateless, horizontally scalable                         │
│                                                              │
│  File: moderator_api.py                                     │
└──────────────────────┬──────────────────────────────────────┘
                       │ Unix Socket
                       ▼
              ┌────────────────┐
              │   GOCIAL       │
              │   Server       │
              │                │
              │  Go HTTP calls │
              │  via socket    │
              └────────────────┘
```

## Performance Characteristics

**Cache Hit (27% of requests):**
- FAISS lookup: ~4ms
- API overhead: <1ms (Unix socket)
- **Total: ~5ms**

**Cache Miss (73% of requests):**
- Fingerprint generation: ~50-100ms
- FAISS lookup: ~4ms (no match)
- Fallback to Ollama: ~460ms
- **Total: ~500-600ms**

**Weighted average: ~350ms per request**

## API Endpoints

### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "text_index_loaded": true,
  "image_index_loaded": true,
  "text_vectors": 14421,
  "image_vectors": 8542
}
```

### Check Text
```http
POST /check/text
Content-Type: application/json

{
  "text": "Content to check"
}
```

Response:
```json
{
  "status": "ALLOWED",  // or "DENIED"
  "confidence": 0.92,
  "match_type": "semantic",  // or "exact_hash", or null
  "reason": "Contains hate speech targeting protected group",
  "lookup_time_ms": 4.2
}
```

### Check Image
```http
POST /check/image
Content-Type: application/json

{
  "image_base64": "iVBORw0KGgoAAAANS..."
}
```

Response:
```json
{
  "status": "DENIED",
  "confidence": 0.95,
  "match_type": "semantic",
  "reason": "Nazi imagery",
  "lookup_time_ms": 8.5
}
```

### Statistics
```http
GET /stats
```

Response:
```json
{
  "text_index": {
    "total_vectors": 14421,
    "dimension": 384
  },
  "image_index": {
    "total_vectors": 8542,
    "dimension": 512
  }
}
```

## Installation & Setup

### 1. Install Dependencies

```bash
pip install fastapi uvicorn requests-unixsocket
```

### 2. Build FAISS Indices

First, make sure you have fingerprints in the database:

```bash
# Scrape content
python scraper.py

# Label content
python gemini_labeler.py
python moderate_text_posts.py

# Build FAISS indices
python build_faiss_index.py
```

### 3. Start API Server

**Unix Socket Mode (Recommended):**
```bash
python moderator_api.py --socket /tmp/moderator.sock
```

**TCP Mode (for testing):**
```bash
python moderator_api.py --host 127.0.0.1 --port 8000
```

### 4. Test the API

**Python client:**
```bash
python test_moderator_api.py
```

**Go client:**
```bash
go run gocial_integration_example.go
```

## Gocial Integration

### Go Client Example

```go
package main

import (
    "context"
    "net"
    "net/http"
)

// Create client with Unix socket
client := &http.Client{
    Transport: &http.Transport{
        DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
            return net.Dial("unix", "/tmp/moderator.sock")
        },
    },
}

// Check content before posting
resp, err := client.Post(
    "http://unix/check/text",
    "application/json",
    bytes.NewBuffer(requestBody),
)
```

### Integration in Post Handler

```go
func (s *PostService) CreatePost(content UserContent) error {
    // Check content
    result := s.moderator.CheckText(content.Text)

    if result.Status == "DENIED" {
        return errors.New("content violates policy")
    }

    // Allowed - proceed with post
    return s.repository.Save(content)
}
```

## Deployment Options

### Option 1: Same Machine (Simplest)
- Gocial and API on same server
- Unix socket: `/tmp/moderator.sock`
- ~5ms latency
- **Recommended for start**

### Option 2: Separate Machine (Scalable)
- API on dedicated server
- Switch to TCP/gRPC
- ~10-15ms latency (local network)
- Horizontal scaling possible

### Option 3: Multiple API Instances
- Load balance across multiple API processes
- Each process on different port/socket
- Gocial round-robins requests
- Linear scalability

## Systemd Service (Production)

Create `/etc/systemd/system/content-moderator.service`:

```ini
[Unit]
Description=Content Moderation API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/content-moderator
ExecStart=/usr/bin/python3 moderator_api.py --socket /var/run/moderator.sock
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Start service:
```bash
sudo systemctl enable content-moderator
sudo systemctl start content-moderator
```

## Monitoring & Logs

### Health Check
```bash
curl --unix-socket /tmp/moderator.sock http://localhost/health
```

### Performance Monitoring
```bash
# Check response times
time curl --unix-socket /tmp/moderator.sock \
  -X POST http://localhost/check/text \
  -H "Content-Type: application/json" \
  -d '{"text":"test content"}'
```

### Log Analysis
```bash
# View API logs
journalctl -u content-moderator -f
```

## Security Considerations

1. **Socket Permissions**: Set appropriate permissions on Unix socket
   ```bash
   chmod 660 /tmp/moderator.sock
   chown gocial:gocial /tmp/moderator.sock
   ```

2. **Rate Limiting**: Add rate limiting in API or Gocial
   - Prevent abuse
   - 100 req/sec per user is reasonable

3. **Fail Open vs Fail Closed**:
   - Current: Fail open (allow on API error)
   - Consider: Fail closed for critical moderation

4. **Database Access**: API has read-only access to fingerprint DB

## Troubleshooting

### API won't start
```bash
# Check if socket already exists
ls -la /tmp/moderator.sock

# Remove old socket
rm /tmp/moderator.sock
```

### Connection refused
```bash
# Check API is running
ps aux | grep moderator_api

# Test socket exists
test -S /tmp/moderator.sock && echo "Socket exists"
```

### Slow responses
```bash
# Check FAISS indices loaded
curl --unix-socket /tmp/moderator.sock http://localhost/stats

# Monitor system resources
htop
```

## Performance Tuning

### Multiple Workers
Run multiple API processes with different sockets:

```bash
python moderator_api.py --socket /tmp/moderator-1.sock &
python moderator_api.py --socket /tmp/moderator-2.sock &
python moderator_api.py --socket /tmp/moderator-3.sock &
```

Gocial load balances across them.

### Preload Models
Models are loaded on startup (2-3 seconds). Keep service running to avoid cold starts.

### FAISS Optimization
- Current: IndexFlatIP (exact search, 692x faster than brute force)
- For >1M vectors: Consider IndexIVFFlat (approximate search, even faster)

## Next Steps

1. **Add fallback moderation**: If novel content (no cache hit), optionally call Ollama/Gemini
2. **Queue system**: Novel content queued for admin review and training
3. **Metrics**: Add Prometheus metrics for monitoring
4. **Caching**: Add Redis cache for very hot content
5. **Circuit breaker**: Fail fast if API is overloaded

## Files

- `moderator_api.py` - FastAPI server
- `test_moderator_api.py` - Python test client
- `gocial_integration_example.go` - Go integration example
- `API_INTEGRATION.md` - This document
