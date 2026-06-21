import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from question_feature_mapping import MODEL_FEATURES, CLASS_LABELS
from siddha_rule_engine import (
    calculate_rule_score,
    normalize_scores_to_percent,
    load_knowledge_base
)


DATASET_PATH = "Siddha_dataset_repaired.csv"

DOSHA_TO_CLASS = {
    "Vata": 0,
    "Pitta": 1,
    "Kapha": 2
}


def row_to_text(row):
    parts = []

    for feature in MODEL_FEATURES:
        value = int(row[feature])

        if value == 0:
            parts.append(f"{feature} Vata dry rough thin cold cracked unstable")
        elif value == 1:
            parts.append(f"{feature} Pitta warm hot sharp reddish yellow anger")
        elif value == 2:
            parts.append(f"{feature} Kapha soft smooth heavy stable oily calm")

    return " ".join(parts)


def get_rule_probabilities_for_row(row, knowledge_base):
    text = row_to_text(row)

    raw_scores, _, _ = calculate_rule_score(
        text,
        knowledge_base
    )

    return normalize_scores_to_percent(raw_scores)


def get_ml_probabilities(model, row):
    input_df = pd.DataFrame(
        [row[MODEL_FEATURES].values],
        columns=MODEL_FEATURES
    )

    probs = model.predict_proba(input_df)[0]

    return {
        CLASS_LABELS[i]: float(probs[i]) * 100
        for i in range(len(probs))
    }


def fuse_probabilities(ml_probs, rule_probs, rule_weight, ml_weight):
    fused = {}

    for dosha in ["Vata", "Pitta", "Kapha"]:
        fused[dosha] = (
            rule_weight * rule_probs.get(dosha, 0)
            +
            ml_weight * ml_probs.get(dosha, 0)
        )

    total = sum(fused.values())

    if total > 0:
        fused = {
            dosha: (score / total) * 100
            for dosha, score in fused.items()
        }

    prediction = max(fused, key=fused.get)

    return DOSHA_TO_CLASS[prediction], fused


def evaluate_fusion_weights_loocv():
    df = pd.read_csv(DATASET_PATH)

    X = df[MODEL_FEATURES]
    y = df["Target_Pragriti_Class"].astype(int).values

    knowledge_base = load_knowledge_base()

    weight_pairs = [
        (0.0, 1.0),
        (0.2, 0.8),
        (0.4, 0.6),
        (0.5, 0.5),
        (0.6, 0.4),
        (0.75, 0.25),
        (0.8, 0.2),
        (1.0, 0.0)
    ]

    loo = LeaveOneOut()
    final_results = []

    for rule_weight, ml_weight in weight_pairs:

        y_true_all = []
        y_pred_all = []

        for train_index, test_index in loo.split(X):
            X_train = X.iloc[train_index]
            y_train = y[train_index]

            test_row = df.iloc[test_index[0]]
            true_label = y[test_index[0]]

            model = RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced"
            )

            model.fit(X_train, y_train)

            ml_probs = get_ml_probabilities(model, test_row)
            rule_probs = get_rule_probabilities_for_row(
                test_row,
                knowledge_base
            )

            pred_class, _ = fuse_probabilities(
                ml_probs,
                rule_probs,
                rule_weight,
                ml_weight
            )

            y_true_all.append(true_label)
            y_pred_all.append(pred_class)

        accuracy = accuracy_score(y_true_all, y_pred_all)
        precision = precision_score(
            y_true_all,
            y_pred_all,
            average="macro",
            zero_division=0
        )
        recall = recall_score(
            y_true_all,
            y_pred_all,
            average="macro",
            zero_division=0
        )
        f1 = f1_score(
            y_true_all,
            y_pred_all,
            average="macro",
            zero_division=0
        )

        final_results.append({
            "Rule_Weight": rule_weight,
            "ML_Weight": ml_weight,
            "Accuracy": round(accuracy, 4),
            "Macro_Precision": round(precision, 4),
            "Macro_Recall": round(recall, 4),
            "Macro_F1": round(f1, 4)
        })

    results_df = pd.DataFrame(final_results)

    results_df = results_df.sort_values(
        by=["Macro_F1", "Accuracy"],
        ascending=False
    )

    results_df.to_csv(
        "fusion_weight_testing_loocv_results.csv",
        index=False
    )

    print("\nLOOCV Fusion Weight Testing Results\n")
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]

    print("\nBest Fusion Configuration")
    print(f"Rule Weight : {best['Rule_Weight']}")
    print(f"ML Weight   : {best['ML_Weight']}")
    print(f"Accuracy    : {best['Accuracy']}")
    print(f"Macro F1    : {best['Macro_F1']}")

    print("\nSaved: fusion_weight_testing_loocv_results.csv")


if __name__ == "__main__":
    evaluate_fusion_weights_loocv()