"""Tests for K-Means and DBSCAN scoring, the DBSCAN knee rule and centroid descriptions (plan §9.4).

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from explorer import clustering
from explorer.methods import INPUT_COLUMNS


def frame_from(x: np.ndarray) -> pd.DataFrame:
    """Rows whose model inputs are exactly x (z columns, then a direction one-hot)."""
    df = pd.DataFrame(x[:, :4], columns=INPUT_COLUMNS[:4])
    df["direction"] = np.where(x[:, 4] > 0.5, "in", np.where(x[:, 5] > 0.5, "out", "none"))
    df["scorable"] = True
    return df


def blobs(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.3, (300, 4))
    b = rng.normal(3, 0.3, (300, 4))
    z = np.vstack([a, b])
    d = np.zeros((600, 2))
    d[:300, 0] = 1
    d[300:, 1] = 1
    return np.hstack([z, d])


class ClusteringTests(unittest.TestCase):
    def test_kmeans_distance_grows_away_from_the_clusters(self):
        rows = frame_from(blobs())
        km = clustering.KMeansMethod(2, 1).fit(rows)
        far = frame_from(np.array([[10, 10, 10, 10, 1, 0.0]]))
        self.assertGreater(km.score(far)[0], km.threshold)
        self.assertEqual(len(set(km.cluster(rows))), 2)

    def test_dbscan_scores_new_points_by_the_nearest_core_sample(self):
        rows = frame_from(blobs())
        db = clustering.DBSCANMethod(eps=1.0, min_samples=12).fit(rows)
        near = frame_from(np.array([[0, 0, 0, 0, 1, 0.0]]))
        far = frame_from(np.array([[8, -8, 8, -8, 0, 1.0]]))
        self.assertLess(db.score(near)[0], db.threshold)
        self.assertGreaterEqual(db.score(far)[0], db.threshold)
        self.assertEqual(db.cluster(far)[0], -1)

    def test_knee_is_where_the_curve_bends(self):
        x = blobs()
        eps0 = clustering.knee_distance(x[:, :4], 12)
        within = np.sort(np.linalg.norm(x[:300, :4] - x[:300, :4].mean(axis=0), axis=1))
        self.assertGreater(eps0, 0)
        self.assertLess(eps0, within[-1] * 2)

    def test_descriptions_read_from_the_centroid(self):
        self.assertEqual(clustering.describe_centroid(np.array([0.1, -0.2, 0.0, 0.3, 0.9, 0.1])),
                         "typical for the address; mostly incoming")
        text = clustering.describe_centroid(np.array([2.5, 0, -1.5, 2.0, 0.1, 0.9]))
        self.assertIn("larger than this address's usual amounts", text)
        self.assertIn("a quick follow-up", text)
        self.assertTrue(text.endswith("mostly outgoing"))


if __name__ == "__main__":
    unittest.main()
