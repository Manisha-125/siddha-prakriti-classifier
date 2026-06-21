import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif

# =====================================================
# LOAD DATASET
# =====================================================

DATASET_PATH = "Siddha_dataset_repaired.csv"

df = pd.read_csv(DATASET_PATH)

print("✅ Dataset Loaded")
print("Shape:", df.shape)

# =====================================================
# FEATURES & TARGET
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
# RANDOM FOREST IMPORTANCE
# =====================================================

print("\n🌲 Running Random Forest Analysis...")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X, y)

rf_importance = rf.feature_importances_

rf_df = pd.DataFrame({
    "Feature": FEATURES,
    "RF_Importance": rf_importance
})

rf_df = rf_df.sort_values(
    by="RF_Importance",
    ascending=True
)

# Plot

plt.figure(figsize=(10, 7))

plt.barh(
    rf_df["Feature"],
    rf_df["RF_Importance"]
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")

plt.tight_layout()

plt.savefig(
    "rf_importance.png",
    dpi=300
)

print("✅ Saved rf_importance.png")

# =====================================================
# EXTRA TREES IMPORTANCE
# =====================================================

print("\n🌳 Running Extra Trees Analysis...")

et = ExtraTreesClassifier(
    n_estimators=200,
    random_state=42
)

et.fit(X, y)

et_importance = et.feature_importances_

et_df = pd.DataFrame({
    "Feature": FEATURES,
    "ET_Importance": et_importance
})

et_df = et_df.sort_values(
    by="ET_Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    et_df["Feature"],
    et_df["ET_Importance"]
)

plt.title("Extra Trees Feature Importance")
plt.xlabel("Importance Score")

plt.tight_layout()

plt.savefig(
    "extra_trees_importance.png",
    dpi=300
)

print("✅ Saved extra_trees_importance.png")

# =====================================================
# MUTUAL INFORMATION
# =====================================================

print("\n📈 Running Mutual Information Analysis...")

mi_scores = mutual_info_classif(
    X,
    y,
    random_state=42
)

mi_df = pd.DataFrame({
    "Feature": FEATURES,
    "MI_Score": mi_scores
})

mi_df = mi_df.sort_values(
    by="MI_Score",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    mi_df["Feature"],
    mi_df["MI_Score"]
)

plt.title("Mutual Information Feature Importance")
plt.xlabel("MI Score")

plt.tight_layout()

plt.savefig(
    "mutual_information.png",
    dpi=300
)

print("✅ Saved mutual_information.png")

# =====================================================
# CREATE RANKINGS
# =====================================================

rf_rank = rf_df.sort_values(
    by="RF_Importance",
    ascending=False
)

rf_rank["RF_Rank"] = range(
    1,
    len(rf_rank) + 1
)

et_rank = et_df.sort_values(
    by="ET_Importance",
    ascending=False
)

et_rank["ET_Rank"] = range(
    1,
    len(et_rank) + 1
)

mi_rank = mi_df.sort_values(
    by="MI_Score",
    ascending=False
)

mi_rank["MI_Rank"] = range(
    1,
    len(mi_rank) + 1
)

# =====================================================
# MERGE RANKINGS
# =====================================================

combined = pd.DataFrame({
    "Feature": FEATURES
})

combined = combined.merge(
    rf_rank[["Feature", "RF_Rank"]],
    on="Feature"
)

combined = combined.merge(
    et_rank[["Feature", "ET_Rank"]],
    on="Feature"
)

combined = combined.merge(
    mi_rank[["Feature", "MI_Rank"]],
    on="Feature"
)

combined["Average_Rank"] = (

    combined["RF_Rank"] +
    combined["ET_Rank"] +
    combined["MI_Rank"]

) / 3

combined = combined.sort_values(
    by="Average_Rank"
)

combined.to_csv(
    "combined_feature_ranking.csv",
    index=False
)

print("\n✅ Saved combined_feature_ranking.csv")

# =====================================================
# COMPARISON CHART
# =====================================================

top_features = combined.sort_values(
    by="Average_Rank"
)

plt.figure(figsize=(12, 8))

x = np.arange(len(top_features))

plt.plot(
    x,
    top_features["RF_Rank"],
    marker='o',
    label="Random Forest"
)

plt.plot(
    x,
    top_features["ET_Rank"],
    marker='s',
    label="Extra Trees"
)

plt.plot(
    x,
    top_features["MI_Rank"],
    marker='^',
    label="Mutual Information"
)

plt.xticks(
    x,
    top_features["Feature"],
    rotation=90
)

plt.ylabel("Rank")
plt.xlabel("Features")

plt.title(
    "Feature Ranking Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "feature_ranking_comparison.png",
    dpi=300
)

print(
    "✅ Saved feature_ranking_comparison.png"
)

print("\n🎉 FEATURE ANALYSIS COMPLETED")