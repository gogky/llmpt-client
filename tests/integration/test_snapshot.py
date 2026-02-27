from huggingface_hub import snapshot_download
try:
    path = snapshot_download(repo_id="gpt2", revision="main")
    print("Snapshot path:", path)
except Exception as e:
    print("Error:", e)
