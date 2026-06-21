import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

# =====================================================
# LOAD DATASET
# =====================================================

DATASET_PATH = "Siddha_dataset_repaired.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset Shape:", df.shape)

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

# =====================================================
# TRAIN FINAL MODEL
# =====================================================

print("\nTraining Final Random Forest Model...")

final_model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)

final_model.fit(X, y)

print("Training Complete!")

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    final_model,
    "prakriti_model.pkl"
)

joblib.dump(
    FEATURES,
    "selected_features.pkl"
)

print("Saved:")
print("prakriti_model.pkl")
print("selected_features.pkl")

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': final_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:")
print(importance_df)

importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

# =====================================================
# FEATURE IMPORTANCE PLOT
# =====================================================

plt.figure(figsize=(10,6))

plt.barh(
    importance_df['Feature'],
    importance_df['Importance']
)

plt.xlabel("Importance Score")
plt.ylabel("Feature")

plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300
)

plt.close()

print("\nSaved:")
print("feature_importance.csv")
print("feature_importance.png")

print("\n===================================")
print("FINAL MODEL READY FOR DEPLOYMENT")
print("===================================")