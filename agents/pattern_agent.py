"""Pattern agent: labels each lot's demand as low, medium, high or peak (K-Means, 4 clusters)."""
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from common import PATTERN_MODEL_PATH

COLS = ["occ_rate", "QueueLength", "traffic"]
LEVELS = ["low", "medium", "high", "peak"]


def train_kmeans(train_df, k=4):
    scaler = StandardScaler().fit(train_df[COLS])
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaler.transform(train_df[COLS]))
    # name clusters by their average occupancy (column 0): lowest = "low", highest = "peak"
    order = kmeans.cluster_centers_[:, 0].argsort()
    names = {int(c): name for c, name in zip(order, LEVELS)}
    return {"scaler": scaler, "kmeans": kmeans, "names": names}


class PatternAgent:
    def __init__(self):
        self.model = joblib.load(PATTERN_MODEL_PATH)

    def label_batch(self, batch):
        """{lot: "low" | "medium" | "high" | "peak"}"""
        m = self.model
        clusters = m["kmeans"].predict(m["scaler"].transform(batch[COLS]))
        return {lot: m["names"][int(c)] for lot, c in zip(batch["SystemCodeNumber"], clusters)}
