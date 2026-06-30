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
