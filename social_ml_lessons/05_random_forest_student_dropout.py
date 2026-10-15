"""
LESSON 5 - RANDOM FOREST CLASSIFIER
Question: "Which university students are at risk of dropping out, and which
factors matter most - so support staff can intervene in the first semester?"

Why it matters: Dropping out often means debt without a degree. Many
universities now run early-warning systems; random forests are a common engine.

What a random forest does:
    * A DECISION TREE asks yes/no questions: "Grades below 10? -> Tuition unpaid? -> ..."
      One tree is easy to read but overfits (memorises quirks).
    * A RANDOM FOREST grows hundreds of trees, each on a random sample of students
      and a random subset of questions, then lets them VOTE. The crowd is far more
      reliable than any single tree.
    * Bonus: it handles curved relationships and interactions automatically, and
      tells you which features it relied on most.

Run:  python 05_random_forest_student_dropout.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

os.makedirs("outputs", exist_ok=True)
rng = np.random.default_rng(21)

# ---------------------------------------------------------------------------
# STEP 1. Simulate 3,000 students
# Built-in interaction: working many hours ONLY hurts much if grades are already
# low. Linear models struggle with this; trees catch it naturally.
# "student_id_digit" is pure noise - a good model should ignore it.
# ---------------------------------------------------------------------------
n = 3000
first_sem_grade = np.clip(rng.normal(12, 3, n), 0, 20)      # 0-20 scale
attendance_pct = np.clip(rng.normal(80, 12, n), 20, 100)
work_hours = rng.choice([0, 10, 20, 30], n, p=[0.4, 0.3, 0.2, 0.1])
tuition_up_to_date = rng.binomial(1, 0.85, n)
scholarship = rng.binomial(1, 0.25, n)
first_generation = rng.binomial(1, 0.35, n)
age = rng.integers(18, 40, n)
student_id_digit = rng.integers(0, 10, n)                   # noise

risk = (-1.5
        - 0.35 * (first_sem_grade - 12)
        - 0.04 * (attendance_pct - 80)
        + 0.10 * work_hours * (first_sem_grade < 11)         # the interaction
        - 1.8 * tuition_up_to_date + 1.8
        - 0.7 * scholarship
        + 0.4 * first_generation
        + 0.03 * (age - 18))
dropout = rng.binomial(1, 1 / (1 + np.exp(-risk)))

df = pd.DataFrame({
    "first_sem_grade": first_sem_grade.round(1), "attendance_pct": attendance_pct.round(0),
    "work_hours": work_hours, "tuition_up_to_date": tuition_up_to_date,
    "scholarship": scholarship, "first_generation": first_generation,
    "age": age, "student_id_digit": student_id_digit, "dropout": dropout,
})
print(df.head(), "\n")
print(f"Dropout rate: {df['dropout'].mean():.1%}\n")

features = [c for c in df.columns if c != "dropout"]
X, y = df[features], df["dropout"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

# ---------------------------------------------------------------------------
# STEP 2. Compare ONE deep tree vs a FOREST
# ---------------------------------------------------------------------------
tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
forest = RandomForestClassifier(
    n_estimators=300,        # number of trees
    min_samples_leaf=5,      # each leaf must hold >= 5 students -> less memorising
    class_weight="balanced", # dropouts are the minority; weight them up
    random_state=0,
    n_jobs=-1,
).fit(X_train, y_train)

for name, m in [("Single tree", tree), ("Random forest", forest)]:
    tr = m.score(X_train, y_train)
    te = m.score(X_test, y_test)
    auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
    print(f"{name:<14} train acc {tr:.0%} | test acc {te:.0%} | test AUC {auc:.2f}")
print("Single tree: ~100% on train, much worse on test = overfitting.\n")

print("Forest, detailed report on unseen students:")
print(classification_report(y_test, forest.predict(X_test), target_names=["stays", "drops out"]))

# ---------------------------------------------------------------------------
# STEP 3. Which factors matter?
# Permutation importance: shuffle one column and see how much the model gets
# worse. Big drop = the model really relied on it. (More trustworthy than the
# built-in .feature_importances_, which favours columns with many unique values.)
# ---------------------------------------------------------------------------
perm = permutation_importance(forest, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=0)
importance = pd.Series(perm.importances_mean, index=features).sort_values()
print("Permutation importance (drop in AUC when shuffled):")
print(importance[::-1].round(3).to_string(), "\n")
print("Note 'student_id_digit' is ~0: the model correctly learned it is noise.\n")

# ---------------------------------------------------------------------------
# STEP 4. A support officer's view: the 10 highest-risk students
# ---------------------------------------------------------------------------
test = X_test.copy()
test["risk"] = forest.predict_proba(X_test)[:, 1]
test["actually_dropped"] = y_test
print("Top 10 students to contact first:")
print(test.sort_values("risk", ascending=False).head(10).round(2).to_string(), "\n")

# ---------------------------------------------------------------------------
# STEP 5. Visualise
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].barh(importance.index, importance.values, color="teal")
axes[0].set_title("What drives dropout risk? (permutation importance)")
axes[0].set_xlabel("Drop in AUC when the feature is shuffled")

# Show the interaction the forest discovered
grades = np.linspace(4, 20, 100)
for hours in [0, 30]:
    probe = pd.DataFrame({f: [X[f].median()] * 100 for f in features})
    probe["first_sem_grade"] = grades
    probe["work_hours"] = hours
    probe["tuition_up_to_date"] = 1
    axes[1].plot(grades, forest.predict_proba(probe)[:, 1], label=f"works {hours} h/week")
axes[1].set_xlabel("First-semester grade (0-20)")
axes[1].set_ylabel("Predicted dropout risk")
axes[1].set_title("Work hours hurt mostly when grades are low")
axes[1].legend()
plt.tight_layout()
plt.savefig("outputs/05_random_forest.png", dpi=120)
print("Saved plot -> outputs/05_random_forest.png")

# ---------------------------------------------------------------------------
# SENIOR DATA SCIENTIST NOTES
# * Importance is not causation. "Scholarship" lowering risk does not prove that
#   giving MORE scholarships will cut dropout by that amount - test with a pilot.
# * Use risk scores to OFFER help (tutoring, emergency grants), never to punish
#   or reject applicants. Same model, very different ethics.
# * Watch for data leakage: features known only AFTER dropout (e.g. "withdrew from
#   courses") make a model look brilliant in testing and useless in real life.
#
# TRY THIS
# 1. Set n_estimators=1 then 10, 100, 500. When does test AUC stop improving?
# 2. Remove "first_sem_grade". Which feature becomes most important?
# 3. Replace the forest with LogisticRegression. Does it capture the work-hours interaction?
# ---------------------------------------------------------------------------
