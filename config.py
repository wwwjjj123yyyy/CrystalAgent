"""
Global configuration
"""

from pathlib import Path
import os

# ==========================
# Project Paths
# ==========================

ROOT = Path(__file__).parent

OUTPUT_DIR = ROOT / "output"
LOG_DIR = ROOT / "logs"
MEMORY_DIR = ROOT / "memory"

OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
MEMORY_DIR.mkdir(exist_ok=True)

# ==========================
# LLM
# ==========================

MODEL_NAME = "gpt-5.5"

TEMPERATURE = 0.8

MAX_TOKENS = 4096

# ==========================
# Agent
# ==========================

NUM_GENERATE = 20

TOP_K = 5

MAX_ITER = 10

# ==========================
# Reflection
# ==========================

ENABLE_REFLECTION = True

ENABLE_MEMORY = True
