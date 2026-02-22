# Development Guide

## Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/llmpt-client.git
cd llmpt-client

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

## Project Structure

```
llmpt-client/
├── llmpt/                      # Main package
│   ├── __init__.py             # Entry point, auto-enable logic
│   ├── patch.py                # Monkey patch implementation
│   ├── tracker_client.py       # Tracker API client
│   ├── downloader.py           # P2P downloader (libtorrent)
│   ├── seeder.py               # Background seeding manager
│   ├── torrent_creator.py      # Torrent creation utilities
│   ├── utils.py                # Helper functions
│   └── cli.py                  # CLI tool
├── tests/                      # Unit tests
├── examples/                   # Example scripts
├── docs/                       # Documentation
└── setup.py                    # Installation config
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llmpt --cov-report=html

# Run specific test file
pytest tests/test_basic.py

# Run with verbose output
pytest -v
```

## Code Style

We use:
- **black** for code formatting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black llmpt/ tests/

# Check linting
flake8 llmpt/ tests/

# Type checking
mypy llmpt/
```

## Development Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make changes**
   - Write code
   - Add tests
   - Update documentation

3. **Test your changes**
   ```bash
   pytest
   black llmpt/ tests/
   flake8 llmpt/ tests/
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "Add your feature"
   git push origin feature/your-feature
   ```

5. **Create pull request**

## Testing with Local Tracker

To test with your local tracker server:

```bash
# Start your tracker server (in llmpt directory)
cd ../llmpt
go run cmd/main.go

# In another terminal, test the client
cd ../llmpt-client
export HF_USE_P2P=1
export HF_P2P_TRACKER=http://localhost:8080

python examples/basic_usage.py
```

## Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from llmpt import enable_p2p
enable_p2p()
```

Or via environment variable:

```bash
export LLMPT_LOG_LEVEL=DEBUG
```

## Common Issues

### libtorrent not installing

On Windows:
```bash
# Download pre-built wheel from:
# https://github.com/arvidn/libtorrent/releases
pip install python_libtorrent-2.0.9-cp311-cp311-win_amd64.whl
```

On Linux:
```bash
sudo apt-get install python3-libtorrent
```

### Import errors

Make sure you installed in development mode:
```bash
pip install -e .
```

## Release Process

1. Update version in `setup.py` and `llmpt/__init__.py`
2. Update CHANGELOG.md
3. Create git tag
4. Build and upload to PyPI

```bash
python setup.py sdist bdist_wheel
twine upload dist/*
```
