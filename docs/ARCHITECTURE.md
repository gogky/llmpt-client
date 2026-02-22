# Architecture

## Overview

llmpt-client provides P2P acceleration for HuggingFace Hub downloads by intercepting HTTP download requests and routing them through BitTorrent when possible.

## Components

### 1. Monkey Patch Layer (`patch.py`)

Intercepts `huggingface_hub` download functions:

```
User Code
    ↓
huggingface_hub.hf_hub_download()  ← Patched
    ↓
huggingface_hub.file_download.http_get()  ← Patched
    ↓
P2P Download or HTTP Fallback
```

**Key Functions:**
- `patched_hf_hub_download()`: Captures repo_id, filename, commit_hash
- `patched_http_get()`: Attempts P2P download, falls back to HTTP

**Thread-Local Context:**
Uses `threading.local()` to pass context from high-level to low-level functions without modifying function signatures.

### 2. Tracker Client (`tracker_client.py`)

Communicates with the llmpt tracker server.

**API Endpoints:**
- `GET /api/v1/torrents` - Query for torrent info
- `POST /api/v1/publish` - Register new torrent
- `GET /announce` - BitTorrent announce (handled by libtorrent)

**Data Flow:**
```
Client → Tracker: Query torrent
Tracker → Client: Torrent info (magnet_link, info_hash)
Client → Tracker: Register new torrent
```

### 3. P2P Downloader (`downloader.py`)

Uses libtorrent to download files via P2P.

**Process:**
1. Parse magnet link
2. Create libtorrent session
3. Add torrent
4. Monitor download progress
5. Handle timeout and errors
6. Copy downloaded file to destination

**Hybrid Download Strategy:**
- Start P2P download
- Monitor progress
- If too slow or stalled, fall back to HTTP
- Use whichever completes first

### 4. Torrent Creator (`torrent_creator.py`)

Creates torrents for downloaded files.

**Process:**
1. Calculate optimal piece length based on file size
2. Generate piece hashes
3. Create torrent with tracker URL
4. Generate magnet link
5. Register with tracker

**Piece Length Strategy:**
- <100MB: 256KB
- 100MB-1GB: 1MB
- 1GB-10GB: 4MB
- >10GB: 16MB

### 5. Seeding Manager (`seeder.py`)

Manages background seeding tasks.

**Features:**
- Background threading
- Configurable seed duration
- Manual stop/start control
- Status monitoring

**Implementation:**
- Each seeding task runs in a daemon thread
- Global dictionary tracks active sessions
- Thread-safe with locks

## Data Flow

### First Download (No Torrent)

```
1. User: snapshot_download("gpt2")
2. llmpt: Query tracker for "gpt2" torrent
3. Tracker: No torrent found
4. llmpt: Download via HTTP (original behavior)
5. llmpt: Create torrent from downloaded file
6. llmpt: Register torrent with tracker
7. llmpt: Start seeding in background
8. Return: File path to user
```

### Subsequent Download (Torrent Exists)

```
1. User: snapshot_download("gpt2")
2. llmpt: Query tracker for "gpt2" torrent
3. Tracker: Return magnet_link + peer list
4. llmpt: Start P2P download via libtorrent
5. llmpt: Connect to peers, download pieces
6. llmpt: Verify downloaded file
7. llmpt: Start seeding in background
8. Return: File path to user
```

## Cache Integration

llmpt respects HuggingFace Hub's cache structure:

```
~/.cache/huggingface/hub/
└── models--gpt2/
    ├── blobs/
    │   └── abc123...  ← Downloaded via P2P
    ├── snapshots/
    │   └── commit_hash/
    │       └── config.json → ../../blobs/abc123...
    └── refs/
        └── main
```

Files are downloaded to the same location as HTTP downloads, ensuring compatibility.

## Error Handling

### Network Errors
- Tracker unreachable → Fall back to HTTP
- No peers found → Fall back to HTTP
- Download timeout → Fall back to HTTP

### libtorrent Unavailable
- Disable P2P entirely
- All downloads use HTTP
- Log warning message

### Torrent Creation Fails
- Continue without seeding
- Log error but don't block user

## Performance Considerations

### Memory
- Streaming downloads (no full file in memory)
- libtorrent handles piece buffering
- Minimal overhead from monkey patching

### CPU
- Piece hashing done by libtorrent (C++)
- Python GIL not a bottleneck (I/O bound)

### Disk
- Same cache structure as original
- No duplicate storage
- Temporary files cleaned up

## Security

### Tracker Trust
- Only connect to configured tracker
- Validate torrent metadata
- No DHT by default (prevents tracking)

### File Integrity
- BitTorrent protocol verifies piece hashes
- Info hash ensures correct file
- Compatible with HuggingFace's etag verification

### Privacy
- Peer IP addresses visible to tracker
- No personal information transmitted
- Can use VPN/proxy if needed

## Future Enhancements

1. **DHT Support**: Decentralized peer discovery
2. **Multi-Tracker**: Fallback trackers
3. **Bandwidth Limiting**: Control upload/download rates
4. **Web UI**: Monitor seeding status
5. **Statistics**: Track P2P vs HTTP usage
6. **Smart Routing**: Choose P2P vs HTTP based on speed
