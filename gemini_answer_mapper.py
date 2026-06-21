import os
import json
import re
from google import genai

from question_feature_mapping import QUESTIONNAIRE, OPTION_TO_VALUE, CLASS_LABELS


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Set it before running.")

client = genai.Client(api_key=GEMINI_API_KEY)


def extract_json(text):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("Invalid JSON returned by Gemini.")


def normalize_scores(scores):
    default_scores = {"A": 0.33, "B": 0.34, "C": 0.33}

    if not isinstance(scores, dict):
        return default_scores

    clean = {}
    for opt in ["A", "B", "C"]:
        try:
            clean[opt] = float(scores.get(opt, 0))
        except Exception:
            clean[opt] = 0.0

    total = sum(clean.values())

    if total <= 0:
        return default_scores

    return {opt: round(clean[opt] / total, 4) for opt in ["A", "B", "C"]}


def map_answer_with_gemini(question_number, patient_response):
    """
    Used when patient speaks/types naturally.
    Gemini maps natural language answer to A/B/C.
    """

    if question_number not in QUESTIONNAIRE:
        raise ValueError("Invalid question number.")

    q = QUESTIONNAIRE[question_number]

    feature = q["feature"]
    question = q["question"]
    options = q["options"]

    prompt = f"""
You are a Siddha clinical questionnaire answer-mapping engine.

Map the patient's answer to the closest option A, B, or C.

Feature:
{feature}

Question:
{question}

Options:
A = {options["A"]} → Vata
B = {options["B"]} → Pitta
C = {options["C"]} → Kapha

Patient response:
"{patient_response}"

Rules:
1. Return ONLY valid JSON.
2. No markdown.
3. Give semantic similarity scores for A, B, C.
4. Scores should approximately add up to 1.0.
5. selected_option must be the option with the highest score.
6. Extract important keywords from the patient response.
7. If the answer is unclear, distribute the scores more evenly.

Return exactly this JSON format:
{{
  "option_scores": {{
    "A": 0.0,
    "B": 0.0,
    "C": 0.0
  }},
  "selected_option": "A",
  "keywords": [],
  "reason": ""
}}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    raw_text = response.text.strip()
    result = extract_json(raw_text)

    option_scores = normalize_scores(result.get("option_scores", {}))

    selected_option = max(option_scores, key=option_scores.get)

    value = OPTION_TO_VALUE[selected_option]
    dosha = CLASS_LABELS[value]

    return {
        "question_number": question_number,
        "feature": feature,
        "question": question,
        "patient_response": patient_response,
        "option_scores": option_scores,
        "selected_option": selected_option,
        "selected_option_text": options[selected_option],
        "value": value,
        "dosha": dosha,
        "keywords": result.get("keywords", []),
        "reason": result.get("reason", ""),
        "input_mode": "speech_or_text"
    }


def map_clicked_option(question_number, selected_option):
    """
    Used when patient directly clicks option A/B/C.
    Gemini is not used here.
    """

    if question_number not in QUESTIONNAIRE:
        raise ValueError("Invalid question number.")

    selected_option = selected_option.upper().strip()

    if selected_option not in OPTION_TO_VALUE:
        raise ValueError("Option must be A, B, or C.")

    q = QUESTIONNAIRE[question_number]

    feature = q["feature"]
    question = q["question"]
    options = q["options"]

    value = OPTION_TO_VALUE[selected_option]
    dosha = CLASS_LABELS[value]

    option_scores = {
        "A": 1.0 if selected_option == "A" else 0.0,
        "B": 1.0 if selected_option == "B" else 0.0,
        "C": 1.0 if selected_option == "C" else 0.0
    }

    return {
        "question_number": question_number,
        "feature": feature,
        "question": question,
        "patient_response": options[selected_option],
        "option_scores": option_scores,
        "selected_option": selected_option,
        "selected_option_text": options[selected_option],
        "value": value,
        "dosha": dosha,
        "keywords": [],
        "reason": "Direct option selected by patient.",
        "input_mode": "option_click"
    }


def build_feature_vector(mapped_answers):
    """
    Converts mapped answers into model input vector in correct feature order.
    """

    from question_feature_mapping import MODEL_FEATURES

    feature_values = {}

    for item in mapped_answers:
        feature_values[item["feature"]] = item["value"]

    vector = []

    missing_features = []

    for feature in MODEL_FEATURES:
        if feature in feature_values:
            vector.append(feature_values[feature])
        else:
            vector.append(1)
            missing_features.append(feature)

    return vector, missing_features


if __name__ == "__main__":
    print("\nTesting Gemini speech/text mapping...\n")

    output = map_answer_with_gemini(
        question_number=5,
        patient_response="My skin is usually very dry and rough, especially during cold weather."
    )

    print(json.dumps(output, indent=4, ensure_ascii=False))

    print("\nTesting direct option click mapping...\n")

    clicked = map_clicked_option(
        question_number=5,
        selected_option="A"
    )

    print(json.dumps(clicked, indent=4, ensure_ascii=False))