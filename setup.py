from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="llmpt-client",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="P2P-accelerated download client for HuggingFace Hub models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llmpt-client",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "huggingface-hub>=0.20.0",
        "libtorrent>=2.0.0",
        "requests>=2.28.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llmpt-cli=llmpt.cli:main",
        ],
    },
    keywords="huggingface p2p bittorrent llm model-download",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/llmpt-client/issues",
        "Source": "https://github.com/yourusername/llmpt-client",
    },
)
