"""
LESSON 3 - POLYNOMIAL REGRESSION
Question: "Does life expectancy keep rising forever as countries get richer,
or do the gains level off?"

Why it matters: If the curve flattens, then $1,000 of aid to a poor country buys
far more years of life than $1,000 in a rich one. This pattern is famous in
global health (the "Preston curve", 1975) and shapes how aid is targeted.

What polynomial regression does: a straight line can't bend. Polynomial
regression adds squared and cubed terms so the line CAN curve:
    life = b0 + b1*gdp + b2*gdp^2 + b3*gdp^3
It is still "linear regression" under the hood - we just give it extra
columns (gdp^2, gdp^3) to work with.

The big lesson here is the bias-variance trade-off:
    too simple  (degree 1)  -> UNDERFITS, misses the curve
    just right  (degree 2-3)
    too complex (degree 15) -> OVERFITS, chases noise and goes wild at the edges

Run:  python 03_polynomial_regression_wealth_health.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

os.makedirs("outputs", exist_ok=True)
rng = np.random.default_rng(3)

# ---------------------------------------------------------------------------
# STEP 1. Simulate 60 countries (small samples make overfitting easy to see)
# Life expectancy climbs fast at low income and flattens near ~82 years.
# ---------------------------------------------------------------------------
n = 60
gdp_k = rng.uniform(0.5, 80, n)                        # GDP per person, $ thousands
life = 50 + 32 * (1 - np.exp(-gdp_k / 12)) + rng.normal(0, 2.5, n)

X = gdp_k.reshape(-1, 1)   # scikit-learn wants a 2D table, even with one column
X_train, X_test, y_train, y_test = train_test_split(X, life, test_size=0.3, random_state=1)

# ---------------------------------------------------------------------------
# STEP 2. Fit polynomials of several degrees and compare
# Training error always goes DOWN with more complexity. Test error is the one
# that tells the truth.
# ---------------------------------------------------------------------------
degrees = [1, 2, 3, 15]
models = {}
print(f"{'degree':>6} | {'train error':>11} | {'test error':>10}")
print("-" * 34)
for d in degrees:
    m = make_pipeline(PolynomialFeatures(degree=d), StandardScaler(), LinearRegression())
    m.fit(X_train, y_train)
    models[d] = m
    tr = np.sqrt(mean_squared_error(y_train, m.predict(X_train)))
    te = np.sqrt(mean_squared_error(y_test, m.predict(X_test)))
    print(f"{d:>6} | {tr:>8.2f} yr | {te:>7.2f} yr")
print("(errors are root-mean-squared, in years of life)\n")

# ---------------------------------------------------------------------------
# STEP 3. Answer the question with the degree-3 model
# ---------------------------------------------------------------------------
m = models[3]
for low, high in [(1, 5), (20, 24), (60, 64)]:
    gain = m.predict([[high]])[0] - m.predict([[low]])[0]
    print(f"Raising GDP from ${low}k to ${high}k per person -> about {gain:+.1f} years of life expectancy")
print("Same $4k increase, very different payoff: the curve flattens.\n")

# ---------------------------------------------------------------------------
# STEP 4. Visualise
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(degrees), figsize=(16, 4.5), sharey=True)
grid = np.linspace(0.5, 80, 400).reshape(-1, 1)
for ax, d in zip(axes, degrees):
    ax.scatter(X_train, y_train, s=12, alpha=0.5, label="train")
    ax.scatter(X_test, y_test, s=12, alpha=0.7, marker="x", label="test")
    ax.plot(grid, models[d].predict(grid), color="crimson", linewidth=2)
    ax.set_ylim(40, 95)
    ax.set_title({1: "Degree 1: underfit", 2: "Degree 2", 3: "Degree 3: good",
                  15: "Degree 15: overfit"}[d])
    ax.set_xlabel("GDP per person ($ thousands)")
axes[0].set_ylabel("Life expectancy (years)")
axes[0].legend()
plt.tight_layout()
plt.savefig("outputs/03_polynomial_regression.png", dpi=120)
print("Saved plot -> outputs/03_polynomial_regression.png")

# ---------------------------------------------------------------------------
# SENIOR DATA SCIENTIST NOTES
# * Never extrapolate a polynomial. Ask the degree-3 model about GDP = $150k
#   and it may predict life expectancy going DOWN. Polynomials have no common sense
#   outside the data range.
# * An often better choice: transform the input. life ~ log(gdp) is a single
#   straight line that captures "diminishing returns" with one parameter.
#   Knowing the right transformation beats adding complexity.
# * Use cross-validation (sklearn.model_selection.cross_val_score) to pick the
#   degree instead of one lucky split.
#
# TRY THIS
# 1. Print models[3].predict([[150]]) and models[15].predict([[150]]). Trust them?
# 2. Fit LinearRegression on np.log(X). Compare its test error to degree 3.
# 3. Increase the noise from 2.5 to 6. Which degree wins now?
# ---------------------------------------------------------------------------
