#!/usr/bin/env python3
"""
Central configuration module for Math Explorer.
Resolves paths dynamically via environment variables with relative fallbacks.
"""

import os
from pathlib import Path

# Base directory for the repository
BASE_DIR = Path(__file__).resolve().parent

# Data and Papers directory
DATA_DIR = Path(os.getenv("MATH_EXPLORER_DATA_DIR", BASE_DIR / "data"))
PAPERS_DIR = DATA_DIR / "papers"
CORPUS_DIR = BASE_DIR / "corpus"

# Model paths
DEFAULT_MODEL_PATH = "/Volumes/sdcard/models/limite-1b-violetto"
VIOLETTO_MODEL_PATH = os.getenv("VIOLETTO_MODEL_PATH", DEFAULT_MODEL_PATH)

# Key paths
DEFAULT_KEY_PATH = str(Path.home() / ".typesafe_key")
TYPESAFE_KEY_PATH = os.getenv("TYPESAFE_KEY_PATH", DEFAULT_KEY_PATH)

# Codex CLI
DEFAULT_CODEX_BIN = str(Path.home() / ".local" / "bin" / "codex")
CODEX_BIN = os.getenv("CODEX_BIN", DEFAULT_CODEX_BIN)

# Network / Server settings
SERVER_PORT = int(os.getenv("PORT", "8765"))
SERVER_HOST = os.getenv("HOST", "127.0.0.1")

# Ensure required directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
