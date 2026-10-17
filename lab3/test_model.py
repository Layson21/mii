import unittest
import numpy as np
from model import load_data, fit_ols, predict


class LinearRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()
        cls.result = fit_ols(cls.data)

    def test_dataset_has_82_rows_and_24_features(self):
        self.assertEqual(len(self.data), 82)
        self.assertEqual(len(self.result["feature_names"]), 24)

    def test_residuals_have_zero_mean_approximately(self):
        self.assertAlmostEqual(float(np.mean(self.result["residuals"])), 0.0, places=8)

    def test_r_squared_is_between_zero_and_one(self):
        self.assertGreaterEqual(self.result["r_squared"], 0.0)
        self.assertLessEqual(self.result["r_squared"], 1.0)

    def test_prediction_matches_fitted_values(self):
        np.testing.assert_allclose(predict(self.data, self.result["coefficients"]), self.result["fitted"])


if __name__ == "__main__":
    unittest.main()
