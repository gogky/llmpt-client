import sys
import logging
import threading
import time

# Mock huggingface_hub modules
class MockSubModule:
    pass

class MockHuggingfaceHub:
    file_download = MockSubModule()
    
    @staticmethod
    def hf_hub_download(repo_id, filename, revision="main", **kwargs):
        print(f"[HF_HUB_MOCK] Downloading {repo_id}/{revision} -> {filename}")
        class TempFile:
            def __init__(self, name):
                self.name = name
        
        # simulate HF thread calling http_get
        return MockHuggingfaceHub.file_download.http_get("http://example.com/mock", TempFile(f"/tmp/mock_{filename.replace('/','_')}"))

sys.modules['huggingface_hub'] = MockHuggingfaceHub()
sys.modules['huggingface_hub.file_download'] = MockHuggingfaceHub.file_download

def mock_http_get(url, temp_file, **kwargs):
    print(f"[HTTP_GET_MOCK] Real HTTP request to {url} saved to {temp_file.name}")
    import time
    time.sleep(1)
    with open(temp_file.name, "w") as f:
        f.write("mock http data")

MockHuggingfaceHub.file_download.http_get = mock_http_get

# Add test path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llmpt.patch import apply_patch
from llmpt.p2p_batch import P2PBatchManager

# Mock TrackerClient
class MockTrackerClient:
    def __init__(self, url):
        self.tracker_url = url
    
    def get_torrent_info(self, repo_id, filename, revision):
        print(f"[MockTrackerClient] Getting info for {repo_id} rev {revision}")
        # Need a real libtorrent-compatible magnet link for a tiny file to test, or we just mock libtorrent itself.
        # Since libtorrent is external, we'll configure p2p_batch to fallback to HTTP if magnet fails.
        return None

logging.basicConfig(level=logging.DEBUG)

def test_single_file():
    print("=== Testing Single File ===")
    config = {'tracker_url': 'http://localhost:8080'}
    apply_patch(config)
    
    MockHuggingfaceHub.hf_hub_download("gpt2", "config.json")

def test_snapshot_concurrent():
    print("\n=== Testing Concurrent Snapshot ===")
    files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
    
    def dl_thread(f):
        MockHuggingfaceHub.hf_hub_download("gpt2", f)
        
    threads = []
    for f in files:
        t = threading.Thread(target=dl_thread, args=(f,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()

if __name__ == "__main__":
    test_single_file()
    test_snapshot_concurrent()
