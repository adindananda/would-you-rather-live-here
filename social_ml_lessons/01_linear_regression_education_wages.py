"""
LESSON 1 - LINEAR REGRESSION
Question: "How much does one extra year of education raise a worker's hourly wage?"

Why it matters: Governments decide how much to spend on schools, scholarships and
adult-education programmes. The "return to education" is one of the most studied
numbers in economics, and linear regression is how it is usually estimated.

What linear regression does: it draws the best straight line through the data.
    wage = intercept + slope_1 * education + slope_2 * experience + ...
Each slope answers: "if this one input goes up by 1, how much does the output change,
holding the other inputs fixed?"

Run:  python 01_linear_regression_education_wages.py
"""
import os

import matplotlib
matplotlib.use("Agg")  # draw plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

os.makedirs("outputs", exist_ok=True)
rng = np.random.default_rng(42)  # fixed seed -> same "random" data every run

# ---------------------------------------------------------------------------
# STEP 1. Get data
# We SIMULATE 1,000 workers so we know the true answer and can check whether the
# model finds it. True rule: each year of school adds $1.80/hour, each year of
# experience adds $0.40/hour, plus noise (luck, location, industry...).
# ---------------------------------------------------------------------------
n = 1000
education = rng.integers(8, 21, n)            # 8 to 20 years of schooling
experience = rng.integers(0, 40, n)           # 0 to 39 years working
noise = rng.normal(0, 4, n)                   # everything we did not measure
wage = 2.0 + 1.80 * education + 0.40 * experience + noise

df = pd.DataFrame({"education": education, "experience": experience, "wage": wage})
print("First five workers:")
print(df.head(), "\n")
print("Summary statistics:")
print(df.describe().round(1), "\n")

# ---------------------------------------------------------------------------
# STEP 2. Split into training and test sets
# The model learns from 80% of people and is graded on the 20% it never saw.
# This is how we check it works on NEW people, not just memorises old ones.
# ---------------------------------------------------------------------------
X = df[["education", "experience"]]   # inputs (features)
y = df["wage"]                        # output (target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------------
# STEP 3. Fit the model
# ---------------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("What the model learned:")
print(f"  Intercept:              ${model.intercept_:.2f}")
print(f"  Per year of education:  ${model.coef_[0]:.2f}/hour   (truth: $1.80)")
print(f"  Per year of experience: ${model.coef_[1]:.2f}/hour   (truth: $0.40)\n")

# ---------------------------------------------------------------------------
# STEP 4. Evaluate on unseen people
#   R^2  = share of wage differences the model explains (1.0 = perfect, 0 = useless)
#   MAE  = on average, how many dollars our prediction is off by
# ---------------------------------------------------------------------------
pred = model.predict(X_test)
print("How good is it on people it never saw?")
print(f"  R^2: {r2_score(y_test, pred):.2f}")
print(f"  Mean absolute error: ${mean_absolute_error(y_test, pred):.2f}/hour\n")

# ---------------------------------------------------------------------------
# STEP 5. Use it to answer a policy question
# ---------------------------------------------------------------------------
example = pd.DataFrame({"education": [12, 16], "experience": [10, 10]})
p12, p16 = model.predict(example)
print("Policy view: a worker with 10 years experience")
print(f"  High-school diploma (12 yrs):  ${p12:.2f}/hour")
print(f"  Bachelor's degree   (16 yrs):  ${p16:.2f}/hour")
print(f"  Difference over a 2,000-hour work year: ${(p16 - p12) * 2000:,.0f}\n")

# ---------------------------------------------------------------------------
# STEP 6. Visualise
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(df["education"], df["wage"], alpha=0.25, s=12)
xs = np.arange(8, 21)
line = model.predict(pd.DataFrame({"education": xs, "experience": df["experience"].mean()}))
axes[0].plot(xs, line, color="crimson", linewidth=2.5, label="model (avg experience)")
axes[0].set_xlabel("Years of education")
axes[0].set_ylabel("Hourly wage ($)")
axes[0].set_title("Each dot is a worker; the line is the model")
axes[0].legend()

# A residual is "actual minus predicted". Good models leave a shapeless cloud
# around zero. Curves or funnels here mean a straight line is the wrong shape.
axes[1].scatter(pred, y_test - pred, alpha=0.4, s=12)
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set_xlabel("Predicted wage ($)")
axes[1].set_ylabel("Error: actual - predicted ($)")
axes[1].set_title("Residuals: should look like random noise")

plt.tight_layout()
plt.savefig("outputs/01_linear_regression.png", dpi=120)
print("Saved plot -> outputs/01_linear_regression.png")

# ---------------------------------------------------------------------------
# SENIOR DATA SCIENTIST NOTES
# * Correlation is not causation. In real data, people who study longer may also
#   come from wealthier families. The education slope would then partly measure
#   family background. Economists use tricks (twin studies, school-reform
#   "natural experiments") to separate the two.
# * Real wages are skewed (a few very high earners). Practitioners usually model
#   log(wage), so the slope reads as "% raise per year of school".
#
# TRY THIS
# 1. Change the noise from 4 to 15. Does the slope stay near 1.80? What happens to R^2?
# 2. Drop "experience" from X. Does the education slope change? Why (or why not here)?
# 3. Change n to 30. How much does the estimated slope wobble between seeds?
# ---------------------------------------------------------------------------
