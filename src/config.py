WEIGHTS_DIR = "weights"
MODEL_NAME = "PPE.pt"

CONF_THRESH = 0.5
MODEL_CONF = 0.7
CONTAINMENT_RATIO = 0.5
PERSON_PAD_PX = 10

PERSON_ALIASES = {"human", "person", "people"}

# REQUIRED_CLASSES = {"glove", "head_cover", "mask", "ppe_coverall", "safety_shoes"}
REQUIRED_CLASSES = {"mask"} # Test

CLASS_SYNONYMS = {"ppe_overall": "ppe_coverall"}
