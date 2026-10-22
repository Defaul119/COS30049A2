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

    def test_each_method_has_one_fixed_colour(self):
        colours = list(plots.METHOD_COLOURS.values())
        self.assertEqual(len(colours), len(set(colours)))
        self.assertIn("baseline", plots.METHOD_COLOURS)


if __name__ == "__main__":
    unittest.main()
