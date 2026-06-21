import json
import joblib
import numpy as np

from question_feature_mapping import MODEL_FEATURES, CLASS_LABELS
from gemini_answer_mapper import build_feature_vector


# =====================================================
# LOAD TRAINED MODEL
# =====================================================

model = joblib.load("prakriti_model.pkl")


# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_prakriti(mapped_answers):
    """
    Takes mapped questionnaire answers and predicts Prakriti.
    """

    vector, missing_features = build_feature_vector(mapped_answers)

    import pandas as pd

    input_df = pd.DataFrame([vector], columns=MODEL_FEATURES)

    prediction_value = int(model.predict(input_df)[0])

    probabilities = model.predict_proba(input_df)[0]

    probability_dict = {
        CLASS_LABELS[i]: round(float(probabilities[i]) * 100, 2)
        for i in range(len(probabilities))
    }

    sorted_probs = sorted(
        probability_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

    primary_dosha = sorted_probs[0][0]
    secondary_dosha = sorted_probs[1][0]

    primary_score = sorted_probs[0][1]
    secondary_score = sorted_probs[1][1]

    # Optional dual constitution logic based on probability closeness
    if primary_score - secondary_score <= 15:
        final_result = f"{primary_dosha}-{secondary_dosha}"
    else:
        final_result = primary_dosha

    return {
        "input_vector": vector,
        "missing_features": missing_features,
        "prediction_value": prediction_value,
        "primary_prediction": CLASS_LABELS[prediction_value],
        "final_result": final_result,
        "probabilities": probability_dict,
        "primary_dosha": primary_dosha,
        "secondary_dosha": secondary_dosha
    }


# =====================================================
# TESTING
# =====================================================

if __name__ == "__main__":

    # Sample mapped answers for all 18 features
    sample_answers = [
        {"feature": "Eyes", "value": 1},
        {"feature": "Nose", "value": 1},
        {"feature": "Tongue", "value": 1},
        {"feature": "Ears", "value": 1},
        {"feature": "Skin", "value": 0},
        {"feature": "Nails", "value": 1},
        {"feature": "Hair", "value": 1},
        {"feature": "Facial Structure", "value": 1},
        {"feature": "Lips", "value": 1},
        {"feature": "Jawline", "value": 1},
        {"feature": "Cheeks", "value": 1},
        {"feature": "Neck", "value": 1},
        {"feature": "Shoulders", "value": 1},
        {"feature": "Chest", "value": 1},
        {"feature": "Arms", "value": 1},
        {"feature": "Hands", "value": 1},
        {"feature": "Legs", "value": 1},
        {"feature": "Feet", "value": 1}
    ]

    result = predict_prakriti(sample_answers)

    print(json.dumps(result, indent=4, ensure_ascii=False))