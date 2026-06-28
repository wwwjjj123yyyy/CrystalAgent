from pathlib import Path
import os

ROOT = Path(__file__).parent

OUTPUT_DIR = ROOT / "output"
LOG_DIR = ROOT / "logs"
MEMORY_DIR = ROOT / "memory"

OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
MEMORY_DIR.mkdir(exist_ok=True)

#########################################
# DeepSeek API
#########################################

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

MODEL_NAME = "deepseek-chat"

TEMPERATURE = 0.7

MAX_TOKENS = 4096

#########################################
# Agent
#########################################

NUM_GENERATE = 20

TOP_K = 5

MAX_ITER = 8
