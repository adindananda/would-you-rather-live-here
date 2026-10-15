"""
LESSON 2 - LOGISTIC REGRESSION
Question: "Which households are at risk of food insecurity, so food-aid programmes
can reach them before they go hungry?"

Why it matters: Food banks and welfare offices have limited budgets. A model that
flags at-risk households lets case workers reach out early instead of waiting
for a crisis.

What logistic regression does: the answer is YES/NO, not a number. Logistic
regression squeezes a straight-line score through an S-shaped curve (the
"sigmoid") so the output is a PROBABILITY between 0 and 1.
    P(food insecure) = 1 / (1 + e^-(intercept + b1*income + b2*household_size + ...))

Run:  python 02_logistic_regression_food_insecurity.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)
rng = np.random.default_rng(7)

# ---------------------------------------------------------------------------
# STEP 1. Simulate 2,000 households
# Risk rises with household size and unemployment, falls with income and if
# the household receives benefits.
# ---------------------------------------------------------------------------
n = 2000
income_k = rng.gamma(shape=3, scale=12, size=n)          # yearly income, $ thousands (skewed, like real income)
household_size = rng.integers(1, 8, n)                   # 1 to 7 people
unemployed = rng.binomial(1, 0.12, n)                    # 12% have an unemployed adult
receives_benefits = rng.binomial(1, 0.25, n)

score = 0.2 - 0.09 * income_k + 0.45 * household_size + 1.6 * unemployed - 0.8 * receives_benefits
prob = 1 / (1 + np.exp(-score))                          # the sigmoid
food_insecure = rng.binomial(1, prob)                    # flip a weighted coin per household

df = pd.DataFrame({
    "income_k": income_k.round(1),
    "household_size": household_size,
    "unemployed": unemployed,
    "receives_benefits": receives_benefits,
    "food_insecure": food_insecure,
})
print(df.head(), "\n")
print(f"Share of households food insecure: {df['food_insecure'].mean():.1%}\n")

# ---------------------------------------------------------------------------
# STEP 2. Split. stratify=y keeps the same YES/NO ratio in train and test.
# ---------------------------------------------------------------------------
features = ["income_k", "household_size", "unemployed", "receives_benefits"]
X, y = df[features], df["food_insecure"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

# ---------------------------------------------------------------------------
# STEP 3. Fit
# StandardScaler puts every feature on the same scale (mean 0, std 1) so the
# coefficients can be compared fairly. A pipeline chains the two steps.
# ---------------------------------------------------------------------------
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)

coefs = model.named_steps["logisticregression"].coef_[0]
print("Which factors push risk up (+) or down (-)?  (per 1 standard deviation)")
for name, c in sorted(zip(features, coefs), key=lambda t: -abs(t[1])):
    print(f"  {name:<18} {c:+.2f}   odds x{np.exp(c):.2f}")
print()

# ---------------------------------------------------------------------------
# STEP 4. Evaluate
#   precision = of the households we flagged, how many really were at risk?
#   recall    = of the households really at risk, how many did we catch?
#   AUC       = how well the model ranks risky above safe households (0.5 = coin flip, 1 = perfect)
# ---------------------------------------------------------------------------
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
print(classification_report(y_test, pred, target_names=["secure", "insecure"]))
print(f"AUC: {roc_auc_score(y_test, proba):.2f}\n")

# In welfare work, MISSING a hungry family is usually worse than an extra phone
# call. Lowering the threshold catches more families at the cost of more false alarms.
for threshold in [0.5, 0.3]:
    p = (proba >= threshold).astype(int)
    caught = ((p == 1) & (y_test == 1)).sum() / (y_test == 1).sum()
    flagged = p.mean()
    print(f"Threshold {threshold}: catch {caught:.0%} of at-risk households, flagging {flagged:.0%} of all households")
print()

# ---------------------------------------------------------------------------
# STEP 5. Score a new household
# ---------------------------------------------------------------------------
new = pd.DataFrame([{"income_k": 35, "household_size": 4, "unemployed": 1, "receives_benefits": 0}])
print(f"New household (income $35k, 4 people, unemployed, no benefits): "
      f"{model.predict_proba(new)[0, 1]:.0%} risk")
new["receives_benefits"] = 1
print(f"Same household after enrolling in benefits:                   "
      f"{model.predict_proba(new)[0, 1]:.0%} risk\n")

# ---------------------------------------------------------------------------
# STEP 6. Visualise
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

incomes = np.linspace(0, 120, 200)
for size in [1, 4, 7]:
    grid = pd.DataFrame({"income_k": incomes, "household_size": size,
                         "unemployed": 0, "receives_benefits": 0})
    axes[0].plot(incomes, model.predict_proba(grid)[:, 1], label=f"{size}-person household")
axes[0].set_xlabel("Household income ($ thousands / year)")
axes[0].set_ylabel("Predicted probability of food insecurity")
axes[0].set_title("The S-curve: risk falls as income rises")
axes[0].legend()

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["secure", "insecure"],
                                        ax=axes[1], colorbar=False)
axes[1].set_title("Confusion matrix (threshold 0.5)")

plt.tight_layout()
plt.savefig("outputs/02_logistic_regression.png", dpi=120)
print("Saved plot -> outputs/02_logistic_regression.png")

# ---------------------------------------------------------------------------
# SENIOR DATA SCIENTIST NOTES
# * Accuracy alone is misleading for rare outcomes. If only 5% are at risk, a
#   model that always says "secure" is 95% accurate and helps nobody.
# * Fairness: check error rates separately by neighbourhood, ethnicity, disability.
#   A model can be accurate overall and still systematically miss one group.
# * A model should support case workers, not replace them.
#
# TRY THIS
# 1. Remove "receives_benefits" from features. How much does AUC drop?
# 2. Try thresholds 0.2 and 0.7. Draw the trade-off between "caught" and "flagged".
# 3. Change 0.12 (unemployment rate) to 0.30 to simulate a recession.
# ---------------------------------------------------------------------------
