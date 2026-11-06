import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
DATA = ROOT / "data"

# 1. Load modelling data
df = pd.read_csv(OUT / "modelling_table.csv")

# 2. Create hub profile
hub_profile = (
    df.groupby("hub_id")
    .agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean")
    )
    .reset_index()
    .sort_values("hub_id")
)

hub_profile.to_csv(OUT / "hub_profile.csv", index=False)

# 3. Scale hub profile
profile_features = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(hub_profile[profile_features])

# 4. K-Means separation evidence
separation_rows = []

for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    distances = pairwise_distances(X_scaled)
    same_cluster = []
    different_cluster = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                same_cluster.append(distances[i, j])
            else:
                different_cluster.append(distances[i, j])

    ratio = np.mean(different_cluster) / np.mean(same_cluster)
    sizes = sorted(np.bincount(labels), reverse=True)

    separation_rows.append({
        "K": k,
        "across_over_within": ratio,
        "cluster_sizes": "-".join(map(str, sizes))
    })

pd.DataFrame(separation_rows).to_csv(
    OUT / "part_d_separation.csv", index=False
)

# 5. Final K-Means with K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
hub_profile["cluster"] = kmeans.fit_predict(X_scaled)

cluster_order = (
    hub_profile.groupby("cluster")["delay_rate"]
    .mean()
    .sort_values()
    .index.tolist()
)

segment_names = {
    cluster_order[0]: "Standard",
    cluster_order[1]: "Watch",
    cluster_order[2]: "Escalate"
}

hub_profile["segment"] = hub_profile["cluster"].map(segment_names)

# 6. Add hub details
hubs = pd.read_csv(DATA / "hubs.csv")

hub_segments = hub_profile[["hub_id", "segment"]].merge(
    hubs[["hub_id", "hub_city", "hub_tier"]],
    on="hub_id",
    how="left"
)

hub_segments.to_csv(OUT / "hub_segments.csv", index=False)

# 7. Segment profile
segment_profile = (
    hub_profile.groupby("segment")
    .agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean")
    )
    .reset_index()
)

segment_profile.to_csv(OUT / "segment_profile.csv", index=False)

# 8. Tier vs segment mismatch
expected_segment = {
    1: "Standard",
    2: "Watch",
    3: "Escalate"
}

hub_segments["expected_segment"] = (
    hub_segments["hub_tier"].map(expected_segment)
)

mismatch = hub_segments[
    hub_segments["segment"] != hub_segments["expected_segment"]
]

mismatch.to_csv(OUT / "tier_vs_segment_mismatch.csv", index=False)

print("Hub profile:", hub_profile.shape)
print("\nSeparation evidence:")
print(pd.read_csv(OUT / "part_d_separation.csv"))
print("\nSegment profile:")
print(segment_profile)
print("\nTier mismatches:")
print(mismatch)