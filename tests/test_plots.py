"""Tests that every report figure builds from the data in the repository (plan §7).

Run from the project root:  python -m pytest -q
The DS1 figures use the committed DS1 tables; figures whose inputs are local-only (Elliptic's
rows, by its licence) are built only where those inputs exist.
"""
from __future__ import annotations

import unittest

import matplotlib.pyplot as plt

from explorer import plots


class FigureTests(unittest.TestCase):
    def test_every_available_figure_builds_titled_with_statistics(self):
        plots.style()
        figures = plots.available()
        self.assertIn("eda_feature_distributions", figures)  # DS1 is committed, so its figures always build
        for name, build in figures.items():
            fig, stats = build()
            with self.subTest(figure=name):
                self.assertTrue(fig._suptitle is not None or fig.axes[0].get_title())  # every figure is titled
                self.assertTrue(all({"figure", "statistic", "value", "note"} <= set(s) for s in stats))
            plt.close(fig)

    def test_the_txgpt_size_figure_builds_from_tx_selects_tables(self):
        # Its tables come from tonight's GPU runs, so build it from lm.tx_select's own functions on made-up runs.
        import pandas as pd

        from lm.tx_select import size_table

        runs = pd.DataFrame({"size": ["s"] * 3 + ["m"] * 3 + ["l"] * 3, "seed": [1, 2, 3] * 3,
                             "parameters": [1e6] * 3 + [4e6] * 3 + [1.6e7] * 3,
                             "best_val_loss": [2.1, 2.2, 2.15, 1.9, 1.95, 1.92, 1.97, 2.0, 1.99],
                             "val_loss_all": [2.1, 2.2, 2.15, 1.9, 1.95, 1.92, 1.97, 2.0, 1.99],
                             "test_loss_all": [2.0, 2.1, 2.05, 1.8, 1.85, 1.82, 1.87, 1.9, 1.89]})
        plots.style()
        fig, stats = plots.fig_txgpt_sizes(runs, size_table(runs))
        self.assertEqual(stats[-1]["value"], "m")
        plt.close(fig)

    def test_each_method_has_one_fixed_colour(self):
        colours = list(plots.METHOD_COLOURS.values())
        self.assertEqual(len(colours), len(set(colours)))
        self.assertIn("baseline", plots.METHOD_COLOURS)


if __name__ == "__main__":
    unittest.main()
