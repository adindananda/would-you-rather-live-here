"""
LESSON 6 - K-NEAREST NEIGHBOURS (KNN)
Question: "What is the air pollution (PM2.5) level in neighbourhoods that have
NO monitoring station?"

Why it matters: Air-quality sensors are expensive, and poorer areas often have
fewer of them - exactly where pollution from highways and factories is often
worst. Estimating the gaps helps cities issue health alerts and decide where to
put the next sensor.

What KNN does: the simplest idea in machine learning - "you are like your
neighbours". To estimate a new location, find the K closest locations we DO
know and average them. No equation is learned; the data IS the model.
    KNN regression     -> average of neighbours' numbers   (used here)
    KNN classification -> majority vote of neighbours' labels (shown at the end)

Run:  python 06_knn_air_quality_estimation.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

os.makedirs("outputs", exist_ok=True)
rng = np.random.default_rng(5)


def true_pollution(x, y):
    """Hidden 'reality' of a 20 x 20 km city: a polluted industrial zone in the
    south-west, a busy highway running diagonally, and clean air elsewhere."""
    industrial = 45 * np.exp(-((x - 4) ** 2 + (y - 5) ** 2) / 12)
    highway = 20 * np.exp(-((y - x - 2) ** 2) / 10)
    return 8 + industrial + highway   # 8 = background level (micrograms per m^3)


# ---------------------------------------------------------------------------
# STEP 1. 250 monitoring stations at random spots, with noisy readings
# ---------------------------------------------------------------------------
n = 250
sx, sy = rng.uniform(0, 20, n), rng.uniform(0, 20, n)
pm25 = true_pollution(sx, sy) + rng.normal(0, 3, n)
df = pd.DataFrame({"x_km": sx, "y_km": sy, "pm25": pm25}).round(2)
print(df.head(), "\n")

X, y = df[["x_km", "y_km"]], df["pm25"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# Note: both features are in km, so no scaling needed. If you mixed km with
# e.g. "traffic count", you MUST scale, because KNN is all about distance.

# ---------------------------------------------------------------------------
# STEP 2. Choosing K - the key decision
#   K=1   -> copies the single nearest station: jumpy, sensitive to a broken sensor
#   K=100 -> averages half the city: smooth but blurs the hotspots away
# Cross-validation tries each K on 5 different splits for a fair score.
# ---------------------------------------------------------------------------
ks = [1, 3, 5, 8, 12, 20, 40, 80]
cv_errors = []
for k in ks:
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance")  # closer stations count more
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
    cv_errors.append(-scores.mean())
    print(f"K={k:>3}: average error {cv_errors[-1]:.2f} ug/m3")
best_k = ks[int(np.argmin(cv_errors))]
print(f"\nBest K: {best_k}\n")

model = KNeighborsRegressor(n_neighbors=best_k, weights="distance").fit(X_train, y_train)
print(f"Test error on held-out stations: {mean_absolute_error(y_test, model.predict(X_test)):.2f} ug/m3\n")

# ---------------------------------------------------------------------------
# STEP 3. Estimate unmonitored places
# WHO guideline for yearly PM2.5 is 5 ug/m3; 25+ is clearly unhealthy.
# ---------------------------------------------------------------------------
places = pd.DataFrame({"x_km": [4, 10, 17], "y_km": [5, 12, 3]},
                      index=["School near factories", "Hospital by highway", "Park, east side"])
places["estimated_pm25"] = model.predict(places[["x_km", "y_km"]]).round(1)
places["true_pm25"] = true_pollution(places["x_km"], places["y_km"]).round(1)
print(places.to_string(), "\n")
print("Look at the hospital: the estimate is too low. Its nearest stations sit off")
print("the highway, so KNN averages in cleaner air. This is a COVERAGE GAP - the")
print("answer is not a fancier model, it is a sensor at the hospital.\n")

dist, idx = model.kneighbors(places[["x_km", "y_km"]].iloc[[0]])
print("The school's estimate is built from these stations:")
print(X_train.iloc[idx[0]].assign(pm25=y_train.iloc[idx[0]].values,
                                  distance_km=dist[0].round(2)).to_string(), "\n")

# ---------------------------------------------------------------------------
# STEP 4. Same idea, as a CLASSIFIER: "unhealthy (>=25) or not?"
# ---------------------------------------------------------------------------
clf = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train >= 25)
acc = accuracy_score(y_test >= 25, clf.predict(X_test))
print(f"KNN classifier accuracy for 'unhealthy air' on held-out stations: {acc:.0%}\n")

# ---------------------------------------------------------------------------
# STEP 5. Visualise: truth vs. too-small K vs. best K vs. too-large K
# ---------------------------------------------------------------------------
g = np.linspace(0, 20, 150)
gx, gy = np.meshgrid(g, g)
grid = pd.DataFrame({"x_km": gx.ravel(), "y_km": gy.ravel()})
panels = [("True pollution (unknown in real life)", true_pollution(gx, gy))]
for k, label in [(1, "K=1: noisy"), (best_k, f"K={best_k}: best"), (80, "K=80: over-smoothed")]:
    m = KNeighborsRegressor(n_neighbors=k, weights="distance" if k != 80 else "uniform").fit(X_train, y_train)
    panels.append((label, m.predict(grid).reshape(gx.shape)))

fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
for ax, (title, z) in zip(axes, panels):
    im = ax.contourf(gx, gy, z, levels=np.linspace(0, 80, 33), cmap="magma_r", extend="max")  # same colour scale on every panel
    ax.scatter(X_train["x_km"], X_train["y_km"], s=6, c="white", edgecolors="black", linewidths=0.3)
    ax.set_title(title)
    ax.set_xlabel("km east")
    ax.set_aspect("equal")
axes[0].set_ylabel("km north")
fig.colorbar(im, ax=axes, label="PM2.5 (ug/m3)", shrink=0.85)
plt.savefig("outputs/06_knn.png", dpi=120, bbox_inches="tight")
print("Saved plot -> outputs/06_knn.png  (dots = monitoring stations)")

# ---------------------------------------------------------------------------
# SENIOR DATA SCIENTIST NOTES
# * KNN is only as good as your coverage. In areas with few stations the
#   "nearest" neighbour may be far away and unrepresentative. Report uncertainty.
# * It gets slow and weak with many features ("curse of dimensionality") because
#   in high dimensions everything is far from everything.
# * Geostatisticians use a cousin of KNN called "kriging" for exactly this task.
#
# TRY THIS
# 1. Set n=40 stations. How much worse is the map? Where does it fail most?
# 2. Delete all stations with x_km < 8 from the training set (an under-monitored
#    poor district). What does KNN predict for the school now?
# 3. Compare weights="uniform" vs "distance" in the cross-validation loop.
# ---------------------------------------------------------------------------
