import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer
from question_feature_mapping import MODEL_FEATURES, CLASS_LABELS


DATASET_PATH = "Siddha_dataset_repaired.csv"
MODEL_PATH = "prakriti_model.pkl"

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATASET_PATH)

X = df[MODEL_FEATURES]
y = df["Target_Pragriti_Class"]

class_names = [
    CLASS_LABELS[0],
    CLASS_LABELS[1],
    CLASS_LABELS[2]
]

explainer = LimeTabularExplainer(
    training_data=X.values,
    feature_names=MODEL_FEATURES,
    class_names=class_names,
    mode="classification",
    discretize_continuous=False,
    random_state=42
)


def model_predict_proba(input_array):
    input_df = pd.DataFrame(
        input_array,
        columns=MODEL_FEATURES
    )

    return model.predict_proba(input_df)


def get_global_feature_importance():
    importance_df = pd.DataFrame({
        "feature": MODEL_FEATURES,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    return importance_df.to_dict(orient="records")


def explain_prediction(
    vector,
    save_path="static/lime_contribution_chart.png",
    top_n=10
):
    if not os.path.exists("static"):
        os.makedirs("static")

    input_array = np.array(vector)

    input_df = pd.DataFrame(
        [vector],
        columns=MODEL_FEATURES
    )

    predicted_class = int(model.predict(input_df)[0])
    predicted_label = CLASS_LABELS[predicted_class]

    probabilities = model.predict_proba(input_df)[0]

    probability_dict = {
        CLASS_LABELS[i]: round(float(probabilities[i]) * 100, 2)
        for i in range(len(probabilities))
    }

    exp = explainer.explain_instance(
        data_row=input_array,
        predict_fn=model_predict_proba,
        num_features=len(MODEL_FEATURES),
        labels=[predicted_class]
    )

    lime_list = exp.as_list(label=predicted_class)

    explanation_data = []

    for feature_condition, weight in lime_list:
        explanation_data.append({
            "feature_condition": feature_condition,
            "feature": feature_condition,
            "weight": round(float(weight), 4),
            "impact": "supports predicted class" if weight > 0 else "opposes predicted class"
        })

    explanation_data_sorted = sorted(
        explanation_data,
        key=lambda x: abs(x["weight"]),
        reverse=True
    )

    top_items = explanation_data_sorted[:top_n]

    features = [
        item["feature_condition"]
        for item in top_items
    ]

    weights = [
        item["weight"]
        for item in top_items
    ]

    colors = [
        "#16a34a" if w > 0 else "#dc2626"
        for w in weights
    ]

    plt.figure(figsize=(10, 6))

    plt.barh(
        features,
        weights,
        color=colors
    )

    plt.axvline(
        0,
        color="black",
        linewidth=0.8
    )

    plt.xlabel("LIME contribution weight")
    plt.title(
        f"Patient-specific LIME explanation: {predicted_label}"
    )

    plt.figtext(
        0.5,
        0.01,
        "Green = supports predicted class | Red = opposes predicted class",
        ha="center",
        fontsize=9
    )

    plt.gca().invert_yaxis()
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "probabilities": probability_dict,
        "lime_explanation": explanation_data_sorted,
        "global_feature_importance": get_global_feature_importance(),
        "chart_path": save_path,
        "note": "LIME explains the local Random Forest prediction. Since the dataset is small, global feature importance and patient response mapping should also be considered."
    }


if __name__ == "__main__":

    sample_vector = [
        1, 1, 1, 1, 0, 1,
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1
    ]

    result = explain_prediction(sample_vector)

    print("\nLIME Explanation Generated Successfully\n")
    print("Predicted:", result["predicted_label"])
    print("Probabilities:", result["probabilities"])
    print("Chart:", result["chart_path"])

    print("\nTop LIME Features:\n")
    for item in result["lime_explanation"][:10]:
        print(item)

    print("\nTop Global Feature Importance:\n")
    for item in result["global_feature_importance"][:10]:
        print(item)