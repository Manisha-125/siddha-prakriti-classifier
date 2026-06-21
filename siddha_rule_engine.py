import json
import re
from pathlib import Path


KNOWLEDGE_BASE_PATH = "siddha_knowledge_base.json"

DOSHAS = ["Vata", "Pitta", "Kapha"]


def load_knowledge_base(path=KNOWLEDGE_BASE_PATH):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            "siddha_knowledge_base.json not found. "
            "Create it in the same folder as this file."
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"[^\w\s\u0B80-\u0BFF\u0D00-\u0D7F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def keyword_found(keyword, text):
    keyword = normalize_text(keyword)
    text = normalize_text(text)

    if not keyword:
        return False

    return keyword in text


def calculate_rule_score(patient_text, knowledge_base):
    patient_text = normalize_text(patient_text)

    raw_scores = {
        "Vata": 0.0,
        "Pitta": 0.0,
        "Kapha": 0.0
    }

    matched_keywords = {
        "Vata": [],
        "Pitta": [],
        "Kapha": []
    }

    category_breakdown = {
        "Vata": {},
        "Pitta": {},
        "Kapha": {}
    }

    for dosha in DOSHAS:
        dosha_data = knowledge_base.get(dosha, {})

        for category_name, category_data in dosha_data.items():

            if category_name == "aliases":
                continue

            category_weight = float(category_data.get("weight", 1.0))
            category_breakdown[dosha][category_name] = 0.0

            for language in ["english", "tamil", "malayalam"]:
                keywords = category_data.get(language, [])

                for keyword in keywords:
                    if keyword_found(keyword, patient_text):
                        raw_scores[dosha] += category_weight
                        category_breakdown[dosha][category_name] += category_weight

                        matched_keywords[dosha].append({
                            "keyword": keyword,
                            "category": category_name,
                            "language": language,
                            "weight": category_weight
                        })

    return raw_scores, matched_keywords, category_breakdown


def normalize_scores_to_percent(raw_scores):
    total = sum(raw_scores.values())

    if total == 0:
        return {
            "Vata": 33.33,
            "Pitta": 33.33,
            "Kapha": 33.34
        }

    return {
        dosha: round((score / total) * 100, 2)
        for dosha, score in raw_scores.items()
    }


def predict_by_rule_engine(mapped_answers):
    knowledge_base = load_knowledge_base()

    combined_text_parts = []

    for item in mapped_answers:
        combined_text_parts.append(str(item.get("patient_response", "")))
        combined_text_parts.append(str(item.get("dosha", "")))
        combined_text_parts.append(str(item.get("reason", "")))

        keywords = item.get("keywords", [])
        if isinstance(keywords, list):
            combined_text_parts.extend(keywords)

    combined_text = " ".join(combined_text_parts)

    raw_scores, matched_keywords, category_breakdown = calculate_rule_score(
        combined_text,
        knowledge_base
    )

    probabilities = normalize_scores_to_percent(raw_scores)

    predicted_dosha = max(
        probabilities,
        key=probabilities.get
    )

    return {
        "rule_prediction": predicted_dosha,
        "rule_probabilities": probabilities,
        "rule_raw_scores": raw_scores,
        "matched_keywords": matched_keywords,
        "category_breakdown": category_breakdown
    }


def fuse_ml_and_rule(ml_probabilities, rule_probabilities, rule_weight=0.60, ml_weight=0.40):
    final_scores = {}

    for dosha in DOSHAS:
        ml_score = float(ml_probabilities.get(dosha, 0))
        rule_score = float(rule_probabilities.get(dosha, 0))

        final_scores[dosha] = round(
            (rule_weight * rule_score) + (ml_weight * ml_score),
            2
        )

    total = sum(final_scores.values())

    if total > 0:
        final_scores = {
            dosha: round((score / total) * 100, 2)
            for dosha, score in final_scores.items()
        }

    final_prediction = max(
        final_scores,
        key=final_scores.get
    )

    return {
        "fusion_prediction": final_prediction,
        "fusion_probabilities": final_scores,
        "rule_weight": rule_weight,
        "ml_weight": ml_weight
    }


if __name__ == "__main__":

    sample_answers = [
        {
            "feature": "Skin",
            "patient_response": "My skin is dry, rough and cracked.",
            "dosha": "Vata",
            "value": 0,
            "keywords": ["dry", "rough", "cracked"],
            "reason": "The response indicates Vata dryness."
        },
        {
            "feature": "Eyes",
            "patient_response": "My eyes become reddish in sunlight.",
            "dosha": "Pitta",
            "value": 1,
            "keywords": ["reddish", "sunlight"],
            "reason": "The response indicates Pitta heat."
        }
    ]

    rule_result = predict_by_rule_engine(sample_answers)

    print(json.dumps(rule_result, indent=4, ensure_ascii=False))

    ml_probabilities = {
        "Vata": 11.5,
        "Pitta": 88.0,
        "Kapha": 0.5
    }

    fusion_result = fuse_ml_and_rule(
        ml_probabilities,
        rule_result["rule_probabilities"],
        rule_weight=0.80,
        ml_weight=0.20
    )

    print("\nFusion Result:")
    print(json.dumps(fusion_result, indent=4, ensure_ascii=False))