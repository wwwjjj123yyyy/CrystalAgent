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
# 变异强度
PERTURBATION = {
    "small": 0.03,
    "medium": 0.08,
    "large": 0.15,
}

LATTICE_SCALE = {
    "small": 0.01,
    "medium": 0.03,
    "large": 0.08,
}

LAYER_SHIFT = {
    "small": 0.10,
    "medium": 0.30,
    "large": 0.60,
}

MAX_VACANCY_FRACTION = 0.05

MIN_INTERATOMIC_DISTANCE = 0.8

SPECIES_POOL = {

"Bi":["Sb","As"],

"Sb":["Bi","As"],

"Te":["Se","S"],

"Se":["Te","S"],

"Sn":["Ge","Pb"]

}
FITNESS = {

"stability":0.45,

"novelty":0.30,

"diversity":0.15,

"validity":0.10

}
