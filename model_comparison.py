import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)

# =====================================================
# LOAD DATASET
# =====================================================

DATASET_PATH = "Siddha_dataset_repaired.csv"

df = pd.read_csv(DATASET_PATH)

print("\nDataset Shape:", df.shape)

# =====================================================
# FEATURES
# =====================================================

FEATURES = [
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

TARGET = "Target_Pragriti_Class"

X = df[FEATURES]
y = df[TARGET]

print("\nClass Distribution")
print(y.value_counts())

# =====================================================
# MODELS
# =====================================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

    "SVM": SVC(
        kernel='rbf',
        C=1,
        gamma='scale'
    )

}

# =====================================================
# SPECIFICITY
# =====================================================

def multiclass_specificity(cm):

    n_classes = cm.shape[0]

    specificities = []

    for i in range(n_classes):

        tp = cm[i, i]

        fn = np.sum(cm[i, :]) - tp

        fp = np.sum(cm[:, i]) - tp

        tn = np.sum(cm) - (tp + fn + fp)

        specificity = tn / (tn + fp + 1e-9)

        specificities.append(specificity)

    return np.mean(specificities)

# =====================================================
# STRATIFIED 3-FOLD
# =====================================================

skf = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

results = []

# =====================================================
# MODEL LOOP
# =====================================================

for model_name, model in models.items():

    print("\n" + "="*70)
    print(f"MODEL: {model_name}")
    print("="*70)

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in skf.split(X, y):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(preds)

    # =================================================
    # METRICS
    # =================================================

    accuracy = accuracy_score(
        y_true_all,
        y_pred_all
    )

    balanced_acc = balanced_accuracy_score(
        y_true_all,
        y_pred_all
    )

    precision_macro = precision_score(
        y_true_all,
        y_pred_all,
        average='macro',
        zero_division=0
    )

    recall_macro = recall_score(
        y_true_all,
        y_pred_all,
        average='macro',
        zero_division=0
    )

    f1_macro = f1_score(
        y_true_all,
        y_pred_all,
        average='macro',
        zero_division=0
    )

    precision_weighted = precision_score(
        y_true_all,
        y_pred_all,
        average='weighted',
        zero_division=0
    )

    recall_weighted = recall_score(
        y_true_all,
        y_pred_all,
        average='weighted',
        zero_division=0
    )

    f1_weighted = f1_score(
        y_true_all,
        y_pred_all,
        average='weighted',
        zero_division=0
    )

    cm = confusion_matrix(
        y_true_all,
        y_pred_all
    )

    specificity = multiclass_specificity(cm)

    # =================================================
    # PRINT
    # =================================================

    print(f"\nAccuracy            : {accuracy:.4f}")
    print(f"Balanced Accuracy   : {balanced_acc:.4f}")

    print(f"\nMacro Precision     : {precision_macro:.4f}")
    print(f"Macro Recall        : {recall_macro:.4f}")
    print(f"Macro F1            : {f1_macro:.4f}")

    print(f"\nWeighted Precision  : {precision_weighted:.4f}")
    print(f"Weighted Recall     : {recall_weighted:.4f}")
    print(f"Weighted F1         : {f1_weighted:.4f}")

    print(f"\nSensitivity         : {recall_macro:.4f}")
    print(f"Specificity         : {specificity:.4f}")

    print("\nClassification Report\n")

    print(
        classification_report(
            y_true_all,
            y_pred_all,
            zero_division=0
        )
    )

    # =================================================
    # SAVE CONFUSION MATRIX
    # =================================================

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title(f"{model_name} Confusion Matrix")

    plt.colorbar()

    plt.xticks(
        [0,1,2],
        ['Vata','Pitta','Kapha']
    )

    plt.yticks(
        [0,1,2],
        ['Vata','Pitta','Kapha']
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha='center',
                va='center'
            )

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"\nSaved: {filename}")

    # =================================================
    # STORE RESULTS
    # =================================================

    results.append({

        "Model": model_name,

        "Accuracy": round(accuracy, 4),

        "Balanced_Accuracy": round(
            balanced_acc,
            4
        ),

        "Macro_Precision": round(
            precision_macro,
            4
        ),

        "Macro_Recall": round(
            recall_macro,
            4
        ),

        "Macro_F1": round(
            f1_macro,
            4
        ),

        "Weighted_Precision": round(
            precision_weighted,
            4
        ),

        "Weighted_Recall": round(
            recall_weighted,
            4
        ),

        "Weighted_F1": round(
            f1_weighted,
            4
        ),

        "Specificity": round(
            specificity,
            4
        )
    })

# =====================================================
# FINAL RESULTS
# =====================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Balanced_Accuracy",
    ascending=False
)

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

print("\n")
print("="*80)
print("FINAL MODEL COMPARISON")
print("="*80)

print(results_df)

best_model = results_df.iloc[0]["Model"]

print("\n🏆 BEST MODEL =", best_model)

print("\nSaved:")
print("model_comparison_results.csv")
print("confusion matrix images")