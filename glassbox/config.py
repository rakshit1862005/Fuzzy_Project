"""Central configuration: file paths, dataset description and default settings.

Every experiment reads its defaults from here, so changing a value in this file
changes it everywhere (scripts, tests and the demo app).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "diabetes.csv"
RESULTS_DIR = ROOT / "results"

SEED = 42

# ---------------------------------------------------------------- dataset
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"
CLASS_NAMES = ("Non-diabetic", "Diabetic")

# In this dataset a 0 in these columns is impossible and means "not measured".
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

UNITS = {
    "Pregnancies": "",
    "Glucose": "mg/dL",
    "BloodPressure": "mm Hg",
    "SkinThickness": "mm",
    "Insulin": "µU/mL",
    "BMI": "kg/m²",
    "DiabetesPedigreeFunction": "",
    "Age": "yrs",
}

# 60 % train / 20 % validation / 20 % test (fractions of the full dataset)
TEST_SIZE = 0.20
VAL_SIZE = 0.20

DECISION_THRESHOLD = 0.5

# ---------------------------------------------------------- neural network
# Grid searched with 5-fold cross-validation on the training set.
NN_PARAM_GRID = {
    "mlp__hidden_layer_sizes": [(8,), (16,), (16, 8), (32, 16)],
    "mlp__alpha": [1e-3, 1e-2, 1e-1, 1.0],
}

# ------------------------------------------------------------ fuzzy system
N_FEATURES = 4          # inputs given to the fuzzy system (top-k by importance)
N_TERMS = 3             # linguistic terms per input: Low / Medium / High
N_RULES = 8             # number of Fuzzy C-Means clusters = upper bound on rules
ANFIS_EPOCHS = 300
ANFIS_LR = 0.01
SIMPLIFY_TOLERANCE = 0.01   # largest fidelity drop allowed while pruning rules
FINE_TUNE_EPOCHS = 100      # extra training of the pruned rule base

# "Oracle" sampling: extra synthetic inputs labelled by the neural network.
ORACLE_SAMPLES = 2000
ORACLE_NOISE = 0.15         # noise std as a fraction of each feature's std

# --------------------------------------------------------------- experiments
SWEEP_RULES = [2, 3, 4, 5, 6, 8, 10, 12, 16]
SWEEP_SEEDS = [0, 1, 2, 3, 4]
