"""
LESSON 4 - K-MEANS CLUSTERING
Question: "What different TYPES of neighbourhoods exist in a city, so services
(clinics, buses, job centres) can be designed for each type instead of
one-size-fits-all?"

Important honesty first: K-means is NOT regression. Regression predicts a known
answer (wage, yes/no). K-means is UNSUPERVISED - there is no answer column. It
finds groups of similar rows on its own. You then look at each group and give it
a meaningful name.

How K-means works (in 4 lines):
    1. Drop K random "centre" points.
    2. Assign each neighbourhood to its nearest centre.
    3. Move each centre to the average of its neighbourhoods.
    4. Repeat 2-3 until nothing changes.

Run:  python 04_kmeans_neighbourhood_types.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)
rng = np.random.default_rng(11)

# ---------------------------------------------------------------------------
# STEP 1. Simulate 400 neighbourhoods built from 4 hidden types.
# The algorithm will NOT be told these types - can it rediscover them?
# Columns: median income ($k), % income spent on rent, % unemployed,
#          green space (m^2 per person), minutes to nearest hospital
# ---------------------------------------------------------------------------
hidden_types = {
    #                         income rent% unemp% green  hosp_min
    "Affluent suburb":       (95,    22,   3,     45,    15),
    "Struggling inner-city": (28,    48,   14,    6,     10),
    "Gentrifying district":  (55,    40,   7,     12,    12),
    "Isolated rural":        (35,    25,   9,     120,   45),
}
spread = np.array([8, 3, 1.5, 8, 4])
rows = []
for name, centre in hidden_types.items():
    rows.append(rng.normal(centre, spread, size=(100, 5)))
data = np.clip(np.vstack(rows), 0, None)
cols = ["income_k", "rent_burden_pct", "unemployment_pct", "green_m2", "hospital_min"]
df = pd.DataFrame(data, columns=cols).round(1)
df = df.sample(frac=1, random_state=0).reset_index(drop=True)   # shuffle
print(df.head(), "\n")

# ---------------------------------------------------------------------------
# STEP 2. Scale. K-means measures distance; without scaling, "green_m2"
# (values ~100) would drown out "unemployment_pct" (values ~10).
# ---------------------------------------------------------------------------
X = StandardScaler().fit_transform(df)

# ---------------------------------------------------------------------------
# STEP 3. How many groups? K-means needs you to choose K.
#   inertia    = total distance to centres (always falls; look for the "elbow")
#   silhouette = how well separated groups are (-1 bad ... 1 great; pick the max)
# ---------------------------------------------------------------------------
ks = range(2, 9)
inertias, silhouettes = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, km.labels_))
    print(f"K={k}: silhouette={silhouettes[-1]:.3f}")
best_k = list(ks)[int(np.argmax(silhouettes))]
print(f"\nBest K by silhouette: {best_k}\n")

# ---------------------------------------------------------------------------
# STEP 4. Fit the final model and DESCRIBE each cluster
# This "profile table" is the real deliverable - it's what a city planner reads.
# ---------------------------------------------------------------------------
km = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(X)
df["cluster"] = km.labels_
profile = df.groupby("cluster")[cols].mean().round(1)
profile["n_neighbourhoods"] = df["cluster"].value_counts().sort_index()
print("Cluster profiles (averages):")
print(profile.to_string(), "\n")
print("Now YOU name them. E.g. high rent burden + high unemployment + few parks")
print("-> 'Struggling inner-city' -> priority for job centres and affordable housing.")
print("Long hospital trips + lots of green -> 'Isolated rural' -> mobile clinics.\n")

# ---------------------------------------------------------------------------
# STEP 5. Visualise. We have 5 columns but screens have 2 dimensions.
# PCA squashes 5D down to the 2 most informative directions for plotting.
# ---------------------------------------------------------------------------
coords = PCA(n_components=2).fit_transform(X)
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

axes[0].plot(list(ks), inertias, marker="o")
axes[0].set_title("Elbow method")
axes[0].set_xlabel("K (number of clusters)")
axes[0].set_ylabel("Inertia")

axes[1].plot(list(ks), silhouettes, marker="o", color="darkgreen")
axes[1].set_title("Silhouette score (higher = better)")
axes[1].set_xlabel("K")

sc = axes[2].scatter(coords[:, 0], coords[:, 1], c=df["cluster"], cmap="tab10", s=14)
axes[2].set_title(f"{best_k} neighbourhood types (PCA view)")
axes[2].set_xlabel("PCA component 1")
axes[2].set_ylabel("PCA component 2")
axes[2].legend(*sc.legend_elements(), title="cluster")

plt.tight_layout()
plt.savefig("outputs/04_kmeans.png", dpi=120)
print("Saved plot -> outputs/04_kmeans.png")

# ---------------------------------------------------------------------------
# SENIOR DATA SCIENTIST NOTES
# * K-means always returns K groups, even from pure noise. Clusters are a lens,
#   not a law of nature - sanity-check them with people who know the city.
# * It assumes round, similar-sized blobs. For odd shapes try DBSCAN.
# * Labels like "struggling" can stigmatise communities. Choose names carefully
#   when results are published.
#
# TRY THIS
# 1. Skip the StandardScaler (use df[cols].values). Do the clusters get worse?
# 2. Force K=2. Which hidden types get merged together?
# 3. Increase "spread" (x2). When do the types stop being separable?
# ---------------------------------------------------------------------------
