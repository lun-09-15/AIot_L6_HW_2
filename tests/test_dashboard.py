"""Offline checks for the interactive regression dashboard."""

import pathlib
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


APP = pathlib.Path(__file__).resolve().parents[1] / "app.py"
namespace = {}
exec(APP.read_text(encoding="utf-8").split("st.set_page_config")[0], namespace)


class DashboardTests(unittest.TestCase):
    def test_model_charts_and_guardrails(self):
        rng = np.random.default_rng(42)
        x = np.arange(120, dtype=float)
        frame = pd.DataFrame({
            "id": np.arange(120),
            "income": x,
            "expenses": rng.normal(20, 3, 120),
            "region": ["north", "south"] * 60,
            "target": x * 0.6 + rng.normal(0, 3, 120),
        })
        frame.loc[2, "income"] = np.inf
        frame.loc[5, "target"] = np.nan

        namespace["validate_dataset"](frame)
        self.assertEqual(namespace["selectable_feature_limit"](frame, "target"), 4)
        profile = namespace["data_profile"](frame, "target")
        summary, result, scatter, interval, bundle = namespace["run_regression"](frame, "target", 4, 0.2)
        diagnostics = namespace["diagnostic_charts"](summary, result)
        curve = namespace["feature_count_cv"](bundle)
        curve_charts = namespace["feature_count_charts"](curve)
        value, lower, upper, notices = namespace["predict_one"](bundle, bundle["default_row"])
        self.assertEqual(len(profile), 3)
        self.assertEqual(len(diagnostics), 4)
        self.assertEqual(len(curve_charts), 2)
        self.assertTrue(all(figure.get_size_inches().tolist() == [7.0, 4.4]
                            for figure in [*diagnostics, *curve_charts, scatter, interval]))
        self.assertEqual(summary["rows"], 119)
        self.assertEqual(summary["raw_features"], 3)
        self.assertEqual(set(summary["comparison"]["模型"]), {"單一特徵", "多元特徵"})
        self.assertGreaterEqual(len(curve), 2)
        self.assertTrue(np.isfinite(summary["rmse"]))
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        self.assertTrue(np.isfinite([value, lower, upper]).all())
        self.assertLess(lower, value)
        self.assertGreater(upper, value)
        self.assertEqual(notices, [])
        bad_row = bundle["default_row"].copy()
        bad_row["income"] = "not a number"
        with self.assertRaisesRegex(ValueError, "有限的數字"):
            namespace["predict_one"](bundle, bad_row)
        for figure in [*profile, *diagnostics, *curve_charts, scatter, interval]:
            plt.close(figure)

    def test_bad_inputs_explain_the_problem(self):
        with self.assertRaisesRegex(ValueError, "沒有資料列"):
            namespace["validate_dataset"](pd.DataFrame(columns=["x", "y"]))
        with self.assertRaisesRegex(ValueError, "數值欄位"):
            namespace["validate_dataset"](pd.DataFrame({"a": ["x"], "b": ["y"]}))
        with self.assertRaisesRegex(ValueError, "至少需要 30 筆"):
            namespace["run_regression"](pd.DataFrame({"x": range(20), "y": range(20)}), "y", 1, 0.2)
        with self.assertRaisesRegex(ValueError, "只有一種有效數值"):
            namespace["run_regression"](pd.DataFrame({"x": range(40), "y": [3] * 40}), "y", 1, 0.2)
        with self.assertRaisesRegex(ValueError, "沒有可用的預測特徵"):
            namespace["run_regression"](pd.DataFrame({"id": range(40), "y": range(40)}), "y", 1, 0.2)
        with self.assertRaisesRegex(ValueError, "http 或 https"):
            namespace["read_csv_url"]("file:///private/data.csv")


if __name__ == "__main__":
    unittest.main()
