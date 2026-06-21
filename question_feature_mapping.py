# =====================================================
# FINAL SIDDHA QUESTIONNAIRE MASTER CONFIGURATION
# =====================================================

QUESTIONNAIRE = {

    1: {
        "feature": "Eyes",
        "question": "What is the nature of your eyes?",
        "options": {
            "A": "Grey/blackish, active, unsteady, dry",
            "B": "Yellow to red, sharp",
            "C": "White, calm, steady, moist"
        }
    },

    2: {
        "feature": "Nose",
        "question": "What best describes your nose?",
        "options": {
            "A": "Thin, long, narrow, slightly bent",
            "B": "Medium, sharp, pointed, reddish tip",
            "C": "Broad, round, well-shaped, moist"
        }
    },

    3: {
        "feature": "Tongue",
        "question": "What is the texture of your tongue?",
        "options": {
            "A": "Black/dry/rough",
            "B": "Reddish and dry",
            "C": "Pale with increased salivation"
        }
    },

    4: {
        "feature": "Ears",
        "question": "What is the nature of your ears?",
        "options": {
            "A": "Small, dry, cool, irregular",
            "B": "Medium, warm, reddish",
            "C": "Large, thick, fleshy"
        }
    },

    5: {
        "feature": "Skin",
        "question": "What is the nature of your skin?",
        "options": {
            "A": "Dry, hard, cool",
            "B": "Wrinkled, sensitive, warm",
            "C": "Smooth, soft, oily, cool"
        }
    },

    6: {
        "feature": "Nails",
        "question": "What is the nature of your nails?",
        "options": {
            "A": "Cracked, brittle, hang nails",
            "B": "Red nail bed with bends",
            "C": "Thick and strong"
        }
    },

    7: {
        "feature": "Hair",
        "question": "What is the nature of your hair?",
        "options": {
            "A": "Grey, split, dry",
            "B": "Yellow to red, sharp",
            "C": "Long, thick, waxy"
        }
    },

    8: {
        "feature": "Facial Structure",
        "question": "What best describes your facial structure?",
        "options": {
            "A": "Thin angular face",
            "B": "Sharp facial features",
            "C": "Round full face"
        }
    },

    9: {
        "feature": "Lips",
        "question": "What is the nature of your lips?",
        "options": {
            "A": "Dry, thin, blackish",
            "B": "Soft, moist, reddish",
            "C": "Thick and glossy"
        }
    },

    10: {
        "feature": "Jawline",
        "question": "What best describes your jawline?",
        "options": {
            "A": "Narrow or weak",
            "B": "Sharp and well-defined",
            "C": "Broad and rounded"
        }
    },

    11: {
        "feature": "Cheeks",
        "question": "What best describes your cheeks?",
        "options": {
            "A": "Sunken or hollow",
            "B": "Medium, slightly reddish",
            "C": "Full, plump and smooth"
        }
    },

    12: {
        "feature": "Neck",
        "question": "What is the nature of your neck?",
        "options": {
            "A": "Thin, long, dry",
            "B": "Medium, muscular",
            "C": "Thick and strong"
        }
    },

    13: {
        "feature": "Shoulders",
        "question": "What best describes your shoulders?",
        "options": {
            "A": "Narrow and bony",
            "B": "Moderately developed",
            "C": "Broad and rounded"
        }
    },

    14: {
        "feature": "Chest",
        "question": "What best describes your chest/body build?",
        "options": {
            "A": "Thin and flat",
            "B": "Moderate build",
            "C": "Broad and full"
        }
    },

    15: {
        "feature": "Arms",
        "question": "What best describes your arms?",
        "options": {
            "A": "Thin and long",
            "B": "Medium and defined",
            "C": "Strong and thick"
        }
    },

    16: {
        "feature": "Hands",
        "question": "What best describes your hands and fingers?",
        "options": {
            "A": "Long tapering fingers",
            "B": "Medium length",
            "C": "Short thick fingers"
        }
    },

    17: {
        "feature": "Legs",
        "question": "What best describes your legs/body frame?",
        "options": {
            "A": "Thin, difficulty gaining weight",
            "B": "Medium build",
            "C": "Heavy build, gains weight easily"
        }
    },

    18: {
        "feature": "Feet",
        "question": "What best describes your feet?",
        "options": {
            "A": "Narrow, dry, cracked",
            "B": "Medium, warm, well-shaped",
            "C": "Large, thick, soft"
        }
    }
}

# =====================================================
# MODEL FEATURE ORDER
# MUST MATCH TRAINING ORDER
# =====================================================

MODEL_FEATURES = [
    'Eyes',
    'Nose',
    'Tongue',
    'Ears',
    'Skin',
    'Nails',
    'Hair',
    'Facial Structure',
    'Lips',
    'Jawline',
    'Cheeks',
    'Neck',
    'Shoulders',
    'Chest',
    'Arms',
    'Hands',
    'Legs',
    'Feet'
]

# =====================================================
# OPTION ENCODING
# =====================================================

OPTION_TO_VALUE = {
    "A": 0,   # Vata
    "B": 1,   # Pitta
    "C": 2    # Kapha
}

VALUE_TO_OPTION = {
    0: "A",
    1: "B",
    2: "C"
}

# =====================================================
# PRAKRITI LABELS
# =====================================================

CLASS_LABELS = {
    0: "Vata",
    1: "Pitta",
    2: "Kapha"
}