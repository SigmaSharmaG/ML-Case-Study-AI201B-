"""
E-COMMERCE CUSTOMER SEGMENTATION
K-Means + DBSCAN on RFM data, PCA for dimensionality reduction, t-SNE for visualisation.

Run:  python rfm_segmentation.py
Outputs: rfm_dataset.csv, rfm_with_clusters.csv, fig1..fig7 (PNG), console results.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

sns.set_theme(style="whitegrid")
RNG = 42
np.random.seed(RNG)

# =====================================================================
# STEP 1: DATASET
# =====================================================================
# Real-world option: UCI "Online Retail II" (transactions). Use build_rfm_from_transactions()
# below to convert it. Here we generate a realistic synthetic RFM table (4,000 customers)
# so the project runs offline and the "true" behaviour groups are known.

def build_rfm_from_transactions(df):
    """df columns: CustomerID, InvoiceNo, InvoiceDate, Quantity, UnitPrice"""
    df = df.dropna(subset=["CustomerID"]).copy()
    df = df[(df.Quantity > 0) & (df.UnitPrice > 0)]
    df["Amount"] = df.Quantity * df.UnitPrice
    snapshot = df.InvoiceDate.max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda d: (snapshot - d.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Amount", "sum"),
    )
    return rfm.reset_index()


def generate_rfm(n=4000):
    # (share, recency mean days, frequency mean orders, avg order value)
    groups = {
        "Champions":   (0.12, 15,  25, 90),
        "Loyal":       (0.22, 45,  10, 60),
        "NewCustomer": (0.20, 20,   1.5, 40),
        "AtRisk":      (0.22, 180,  6, 55),
        "Lost":        (0.24, 300,  1.3, 30),
    }
    rows = []
    for g, (share, r_mu, f_mu, aov) in groups.items():
        k = int(n * share)
        r = np.clip(np.random.gamma(6, r_mu / 6, k), 1, 365)
        f = np.clip(np.random.poisson(f_mu, k), 1, None)
        m = f * np.random.lognormal(np.log(aov), 0.25, k)
        rows.append(pd.DataFrame({"Recency": r.round(), "Frequency": f, "Monetary": m.round(2)}))
    df = pd.concat(rows, ignore_index=True)
    # a few extreme "wholesale" buyers -> natural outliers (good for DBSCAN)
    whales = pd.DataFrame({
        "Recency": np.random.randint(1, 30, 25),
        "Frequency": np.random.randint(60, 120, 25),
        "Monetary": np.random.uniform(15000, 40000, 25).round(2),
    })
    df = pd.concat([df, whales], ignore_index=True).sample(frac=1, random_state=RNG).reset_index(drop=True)
    df.insert(0, "CustomerID", np.arange(10001, 10001 + len(df)))
    return df

rfm = generate_rfm()
rfm.to_csv("rfm_dataset.csv", index=False)
print("=== STEP 1: DATASET ===")
print(rfm.head(), "\n")
print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(2), "\n")

# =====================================================================
# STEP 2: EDA + PREPROCESSING
# =====================================================================
# RFM data is heavily right-skewed -> log transform, then scale (K-Means & DBSCAN use distance).
fig, ax = plt.subplots(2, 3, figsize=(14, 7))
for i, c in enumerate(["Recency", "Frequency", "Monetary"]):
    sns.histplot(rfm[c], bins=40, ax=ax[0, i], color="tomato").set_title(f"{c} (raw, skewed)")
    sns.histplot(np.log1p(rfm[c]), bins=40, ax=ax[1, i], color="seagreen").set_title(f"log({c}) (after transform)")
plt.tight_layout(); plt.savefig("fig1_distributions.png", dpi=130); plt.close()

X_log = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
X = StandardScaler().fit_transform(X_log)

# =====================================================================
# STEP 3: PCA
# =====================================================================
pca_full = PCA().fit(X)
ev = pca_full.explained_variance_ratio_
print("=== STEP 3: PCA ===")
print("Explained variance per component:", ev.round(3))
print("Cumulative:", ev.cumsum().round(3))
print("Loadings (rows = PCs, cols = R, F, M):")
print(pd.DataFrame(pca_full.components_, columns=["Recency", "Frequency", "Monetary"],
                   index=["PC1", "PC2", "PC3"]).round(3), "\n")

pca = PCA(n_components=2, random_state=RNG)
X_pca = pca.fit_transform(X)

# =====================================================================
# STEP 4: K-MEANS  (choose k with elbow + silhouette)
# =====================================================================
ks = range(2, 10)
inertia, sil, dbi = [], [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RNG).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))
    dbi.append(davies_bouldin_score(X, km.labels_))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set(title="Elbow (inertia)", xlabel="k")
ax[1].plot(ks, sil, "o-", color="green"); ax[1].set(title="Silhouette (higher = better)", xlabel="k")
ax[2].plot(ks, dbi, "o-", color="purple"); ax[2].set(title="Davies-Bouldin (lower = better)", xlabel="k")
plt.tight_layout(); plt.savefig("fig2_choose_k.png", dpi=130); plt.close()

print("=== STEP 4: K-MEANS ===")
print(pd.DataFrame({"k": list(ks), "inertia": np.round(inertia, 1),
                    "silhouette": np.round(sil, 3), "davies_bouldin": np.round(dbi, 3)}).to_string(index=False))

K = 5   # chosen: business-interpretable AND silhouette is competitive (see report)
km = KMeans(n_clusters=K, n_init=10, random_state=RNG).fit(X)
rfm["KMeans"] = km.labels_
print(f"\nChosen k = {K}; silhouette = {silhouette_score(X, km.labels_):.3f}\n")

# =====================================================================
# STEP 5: DBSCAN  (choose eps with k-distance graph)
# =====================================================================
min_samples = 6   # rule of thumb: 2 x dimensions (=6)
nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
kdist = np.sort(nn.kneighbors(X)[0][:, -1])
plt.figure(figsize=(7, 4)); plt.plot(kdist)
plt.axhline(0.35, color="r", ls="--", label="eps = 0.35")
plt.title(f"k-distance graph (k={min_samples}) -> pick eps at the 'knee'")
plt.ylabel("distance to k-th neighbour"); plt.xlabel("points sorted"); plt.legend()
plt.tight_layout(); plt.savefig("fig3_kdistance.png", dpi=130); plt.close()

EPS = 0.35
db = DBSCAN(eps=EPS, min_samples=min_samples).fit(X)
rfm["DBSCAN"] = db.labels_
n_cl = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
n_noise = int((db.labels_ == -1).sum())
print("=== STEP 5: DBSCAN ===")
print(f"eps={EPS}, min_samples={min_samples} -> clusters found: {n_cl}, noise points: {n_noise} "
      f"({n_noise / len(rfm):.1%})")
mask = db.labels_ != -1
if n_cl > 1:
    print(f"Silhouette (excluding noise): {silhouette_score(X[mask], db.labels_[mask]):.3f}")
print(rfm["DBSCAN"].value_counts().sort_index().to_string(), "\n")

# =====================================================================
# STEP 6: VISUALISE  (PCA 2D and t-SNE 2D)
# =====================================================================
tsne = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto", random_state=RNG)
X_tsne = tsne.fit_transform(X)

def scatter(ax, emb, labels, title):
    pal = sns.color_palette("tab10", len(set(labels)))
    for i, l in enumerate(sorted(set(labels))):
        m = labels == l
        name = "Noise" if l == -1 else f"Cluster {l}"
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.6,
                   c="black" if l == -1 else [pal[i]], label=name)
    ax.set_title(title); ax.legend(markerscale=2, fontsize=8)

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
scatter(ax[0], X_pca, rfm["KMeans"].values, f"PCA (2D) - K-Means  [{ev[:2].sum():.0%} variance kept]")
scatter(ax[1], X_tsne, rfm["KMeans"].values, "t-SNE (2D) - K-Means")
plt.tight_layout(); plt.savefig("fig4_kmeans_pca_tsne.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
scatter(ax[0], X_pca, rfm["DBSCAN"].values, "PCA (2D) - DBSCAN")
scatter(ax[1], X_tsne, rfm["DBSCAN"].values, "t-SNE (2D) - DBSCAN")
plt.tight_layout(); plt.savefig("fig5_dbscan_pca_tsne.png", dpi=130); plt.close()

# =====================================================================
# STEP 7: PROFILE CLUSTERS -> PERSONAS
# =====================================================================
prof = rfm.groupby("KMeans").agg(
    Customers=("CustomerID", "count"),
    Recency=("Recency", "mean"), Frequency=("Frequency", "mean"),
    Monetary=("Monetary", "mean"), TotalRevenue=("Monetary", "sum"),
).round(1)
prof["Share_of_customers_%"] = (prof.Customers / len(rfm) * 100).round(1)
prof["Share_of_revenue_%"] = (prof.TotalRevenue / prof.TotalRevenue.sum() * 100).round(1)

# Persona naming: simple, explainable business rules on the cluster averages
def name_persona(row):
    if row.Frequency > 20:                          return "Champions"
    if row.Recency < 100 and row.Frequency >= 5:    return "Loyal Regulars"
    if row.Recency < 100:                           return "New / Promising"
    if row.Frequency >= 3:                          return "At-Risk Customers"
    return "Lost / Hibernating"

prof["Persona"] = prof.apply(name_persona, axis=1)
print("=== STEP 7: CLUSTER PROFILES (K-Means) ===")
print(prof.to_string(), "\n")

persona_map = prof["Persona"].to_dict()
rfm["Persona"] = rfm["KMeans"].map(persona_map)

# Heatmap: relative importance of each metric per cluster
rel = (rfm.groupby("KMeans")[["Recency", "Frequency", "Monetary"]].mean()
       / rfm[["Recency", "Frequency", "Monetary"]].mean() - 1)
plt.figure(figsize=(7, 4))
sns.heatmap(rel, annot=True, fmt=".2f", cmap="RdYlGn_r", center=0)
plt.title("Cluster mean vs overall mean (0 = average)")
plt.tight_layout(); plt.savefig("fig6_heatmap.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, c in enumerate(["Recency", "Frequency", "Monetary"]):
    sns.boxplot(data=rfm, x="KMeans", y=c, ax=ax[i], palette="tab10", hue="KMeans", legend=False)
    if c != "Recency": ax[i].set_yscale("log")
    ax[i].set_title(f"{c} by cluster")
plt.tight_layout(); plt.savefig("fig7_boxplots.png", dpi=130); plt.close()

# DBSCAN noise = the extreme wholesale buyers?
noise = rfm[rfm.DBSCAN == -1]
print("DBSCAN noise points - average RFM:")
print(noise[["Recency", "Frequency", "Monetary"]].mean().round(1).to_string(), "\n")

# K-Means vs DBSCAN agreement
from sklearn.metrics import adjusted_rand_score
print(f"Adjusted Rand Index (KMeans vs DBSCAN): {adjusted_rand_score(rfm.KMeans, rfm.DBSCAN):.3f}")

rfm.to_csv("rfm_with_clusters.csv", index=False)
prof.to_csv("cluster_profiles.csv")
print("\nDone. Files saved.")
