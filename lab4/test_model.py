import unittest
import numpy as np
from model import load_data, TwoLayerPerceptron, regression_metrics


class PerceptronTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_data()
        cls.x = cls.frame.drop(columns=["Вещество", "logP"]).to_numpy(dtype=float)
        cls.y = cls.frame["logP"].to_numpy(dtype=float)

    def test_supported_hidden_sizes(self):
        for size in (2, 4, 8, 16, 32):
            self.assertEqual(TwoLayerPerceptron(hidden_size=size).hidden_size, size)

    def test_invalid_hidden_size_is_rejected(self):
        with self.assertRaises(ValueError):
            TwoLayerPerceptron(hidden_size=3)

    def test_training_produces_finite_predictions(self):
        model = TwoLayerPerceptron(hidden_size=4, max_iter=1000).fit(self.x, self.y)
        predictions = model.predict(self.x)
        self.assertEqual(predictions.shape, self.y.shape)
        self.assertTrue(np.isfinite(predictions).all())

    def test_metrics_are_consistent(self):
        metrics = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(metrics["mse"], 0.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)


if __name__ == "__main__":
    unittest.main()
