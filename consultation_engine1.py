import json

from gemini_answer_mapper import build_feature_vector
from predict_prakriti import predict_prakriti
from lime_explainer import explain_prediction

from siddha_rule_engine import (
    predict_by_rule_engine,
    fuse_ml_and_rule
)


RULE_WEIGHT = 0.80
ML_WEIGHT = 0.20


def run_consultation(mapped_answers):
    """
    Hybrid Siddha Consultation Pipeline

    Gemini Mapped Answers
            ↓
    Feature Vector
            ↓
    Random Forest ML Prediction
            ↓
    Noi Naadal Rule Engine Prediction
            ↓
    Weighted Late Fusion 80:20
            ↓
    Final Hybrid Result + LIME + Explanation
    """

    # ==========================
    # 1. Build Feature Vector
    # ==========================
    vector, missing_features = build_feature_vector(mapped_answers)

    # ==========================
    # 2. ML Prediction
    # ==========================
    ml_result = predict_prakriti(mapped_answers)

    ml_probabilities = {
        "Vata": ml_result["probabilities"].get("Vata", 0),
        "Pitta": ml_result["probabilities"].get("Pitta", 0),
        "Kapha": ml_result["probabilities"].get("Kapha", 0)
    }

    # ==========================
    # 3. Noi Naadal Rule Engine
    # ==========================
    rule_result = predict_by_rule_engine(mapped_answers)

    rule_probabilities = rule_result["rule_probabilities"]

    # ==========================
    # 4. Weighted Late Fusion
    # ==========================
    fusion_result = fuse_ml_and_rule(
        ml_probabilities=ml_probabilities,
        rule_probabilities=rule_probabilities,
        rule_weight=RULE_WEIGHT,
        ml_weight=ML_WEIGHT
    )

    final_prediction = fusion_result["fusion_prediction"]
    fusion_probabilities = fusion_result["fusion_probabilities"]

    # ==========================
    # 5. LIME Explanation
    # ==========================
    lime_result = explain_prediction(vector)

    # ==========================
    # 6. Question-wise Table
    # ==========================
    table_data = []

    for i, item in enumerate(mapped_answers, start=1):
        table_data.append({
            "index": i,
            "feature": item.get("feature", "N/A"),
            "response": item.get("patient_response", "N/A"),
            "selected_option": item.get("selected_option", "N/A"),
            "value": item.get("value", "N/A"),
            "dosha": item.get("dosha", "N/A"),
            "keywords": item.get("keywords", []),
            "reason": item.get("reason", "N/A"),
            "input_mode": item.get("input_mode", "N/A"),
            "statement": (
                f"{item.get('feature', 'Feature')} suggests "
                f"{item.get('dosha', 'N/A')} characteristics."
            )
        })

    # ==========================
    # 7. Matched Keyword Summary
    # ==========================
    matched_keyword_summary = []

    for dosha, keywords in rule_result["matched_keywords"].items():
        for kw in keywords:
            matched_keyword_summary.append({
                "dosha": dosha,
                "keyword": kw.get("keyword", ""),
                "category": kw.get("category", ""),
                "language": kw.get("language", ""),
                "weight": kw.get("weight", "")
            })

    matched_keyword_summary = sorted(
        matched_keyword_summary,
        key=lambda x: float(x["weight"]) if x["weight"] != "" else 0,
        reverse=True
    )

    # ==========================
    # 8. Clinical Narrative
    # ==========================
    explanation_text = (
        f"The final assessment indicates a dominant {final_prediction} constitution. "
        f"This result was obtained using a hybrid weighted late-fusion approach. "
        f"The Random Forest machine learning model predicted {ml_result['primary_prediction']} "
        f"with probabilities Vata {ml_probabilities['Vata']}%, "
        f"Pitta {ml_probabilities['Pitta']}%, and Kapha {ml_probabilities['Kapha']}%. "
        f"The Noi Naadal rule-based knowledge engine predicted "
        f"{rule_result['rule_prediction']} with probabilities "
        f"Vata {rule_probabilities['Vata']}%, "
        f"Pitta {rule_probabilities['Pitta']}%, and "
        f"Kapha {rule_probabilities['Kapha']}%. "
        f"Using the experimentally selected fusion ratio of "
        f"{int(RULE_WEIGHT * 100)}% rule engine and {int(ML_WEIGHT * 100)}% ML model, "
        f"the final fused prediction became {final_prediction}. "
        f"The LIME explanation shows the local Random Forest feature contribution, "
        f"while the Noi Naadal keyword matches provide Siddha literature-based interpretability."
    )

    # ==========================
    # 9. Final Report Object
    # ==========================
    report = {
        "prediction": final_prediction,

        "v_prob": fusion_probabilities.get("Vata", 0),
        "p_prob": fusion_probabilities.get("Pitta", 0),
        "k_prob": fusion_probabilities.get("Kapha", 0),

        "ml_prediction": ml_result["primary_prediction"],
        "ml_probabilities": ml_probabilities,

        "rule_prediction": rule_result["rule_prediction"],
        "rule_probabilities": rule_probabilities,
        "rule_raw_scores": rule_result["rule_raw_scores"],
        "matched_keywords": matched_keyword_summary,
        "category_breakdown": rule_result["category_breakdown"],

        "fusion_prediction": final_prediction,
        "fusion_probabilities": fusion_probabilities,
        "rule_weight": RULE_WEIGHT,
        "ml_weight": ML_WEIGHT,

        "table_data": table_data,

        "lime_data": lime_result["lime_explanation"],
        "global_feature_importance": lime_result["global_feature_importance"],
        "chart_path": lime_result["chart_path"],

        "explanation": explanation_text,
        "missing_features": missing_features
    }

    return report


if __name__ == "__main__":

    sample_answers = [
        {"feature": "Eyes", "patient_response": "White, calm, steady, moist", "selected_option": "C", "dosha": "Kapha", "value": 2, "keywords": ["white", "calm", "steady", "moist"], "reason": "Kapha eye features.", "input_mode": "option_click"},
        {"feature": "Nose", "patient_response": "Medium, sharp, pointed, reddish tip", "selected_option": "B", "dosha": "Pitta", "value": 1, "keywords": ["sharp", "reddish"], "reason": "Pitta nose features.", "input_mode": "option_click"},
        {"feature": "Tongue", "patient_response": "Black, dry, rough", "selected_option": "A", "dosha": "Vata", "value": 0, "keywords": ["black", "dry", "rough"], "reason": "Vata tongue features.", "input_mode": "option_click"},
        {"feature": "Ears", "patient_response": "Medium, warm, reddish", "selected_option": "B", "dosha": "Pitta", "value": 1, "keywords": ["warm", "reddish"], "reason": "Pitta ear features.", "input_mode": "option_click"},
        {"feature": "Skin", "patient_response": "Dry, hard, cool", "selected_option": "A", "dosha": "Vata", "value": 0, "keywords": ["dry", "hard", "cool"], "reason": "Vata skin features.", "input_mode": "option_click"},
        {"feature": "Nails", "patient_response": "Cracked, brittle, hang nails", "selected_option": "A", "dosha": "Vata", "value": 0, "keywords": ["cracked", "brittle"], "reason": "Vata nail features.", "input_mode": "option_click"},
        {"feature": "Hair", "patient_response": "Grey, split, dry", "selected_option": "A", "dosha": "Vata", "value": 0, "keywords": ["grey", "split", "dry"], "reason": "Vata hair features.", "input_mode": "option_click"},
        {"feature": "Facial Structure", "patient_response": "Thin angular face", "selected_option": "A", "dosha": "Vata", "value": 0, "keywords": ["thin", "angular"], "reason": "Vata facial features.", "input_mode": "option_click"},
        {"feature": "Lips", "patient_response": "Dry, thin, blackish", "selected_option": "A", "dosha": "Vata", "value": 0, "keywords": ["dry", "thin", "blackish"], "reason": "Vata lip features.", "input_mode": "option_click"},
        {"feature": "Jawline", "patient_response": "Sharp and well-defined", "selected_option": "B", "dosha": "Pitta", "value": 1, "keywords": ["sharp"], "reason": "Pitta jawline features.", "input_mode": "option_click"},
        {"feature": "Cheeks", "patient_response": "Full, plump and smooth", "selected_option": "C", "dosha": "Kapha", "value": 2, "keywords": ["full", "plump", "smooth"], "reason": "Kapha cheek features.", "input_mode": "option_click"},
        {"feature": "Neck", "patient_response": "Medium, muscular", "selected_option": "B", "dosha": "Pitta", "value": 1, "keywords": ["medium", "muscular"], "reason": "Pitta neck features.", "input_mode": "option_click"},
        {"feature": "Shoulders", "patient_response": "Narrow and bony", "selected_option": "A", "dosha": "Vata", "value": 0, "keywords": ["narrow", "bony"], "reason": "Vata shoulder features.", "input_mode": "option_click"},
        {"feature": "Chest", "patient_response": "Moderate build", "selected_option": "B", "dosha": "Pitta", "value": 1, "keywords": ["moderate"], "reason": "Pitta chest features.", "input_mode": "option_click"},
        {"feature": "Arms", "patient_response": "Medium and defined", "selected_option": "B", "dosha": "Pitta", "value": 1, "keywords": ["medium", "defined"], "reason": "Pitta arm features.", "input_mode": "option_click"},
        {"feature": "Hands", "patient_response": "Medium length", "selected_option": "B", "dosha": "Pitta", "value": 1, "keywords": ["medium"], "reason": "Pitta hand features.", "input_mode": "option_click"},
        {"feature": "Legs", "patient_response": "Thin, difficulty gaining weight", "selected_option": "A", "dosha": "Vata", "value": 0, "keywords": ["thin"], "reason": "Vata leg features.", "input_mode": "option_click"},
        {"feature": "Feet", "patient_response": "Medium, warm, well-shaped", "selected_option": "B", "dosha": "Pitta", "value": 1, "keywords": ["medium", "warm"], "reason": "Pitta feet features.", "input_mode": "option_click"}
    ]

    result = run_consultation(sample_answers)

    print(json.dumps(result, indent=4, ensure_ascii=False))