import unittest

import numpy as np

from lab2_solution import (
    bernoulli_2x2_analytic,
    eigenvalue_spacing,
    generate_matrix,
    normalize_spacings,
)


class Lab2Tests(unittest.TestCase):
    def test_generate_normal_matrix_is_symmetric(self):
        rng = np.random.default_rng(1)
        matrix = generate_matrix(4, 'normal', rng)
        self.assertEqual(matrix.shape, (4, 4))
        self.assertTrue(np.allclose(matrix, matrix.T))

    def test_eigenvalue_spacing_for_diagonal_matrix(self):
        matrix = np.diag([1.0, 3.0, 7.0, 10.0])
        spacing = eigenvalue_spacing(matrix)
        self.assertAlmostEqual(spacing, 4.0)

    def test_normalize_spacings(self):
        spacings = np.array([1.0, 2.0, 3.0, 4.0])
        normalized, mean_spacing = normalize_spacings(spacings)
        self.assertAlmostEqual(mean_spacing, 2.5)
        self.assertAlmostEqual(float(np.mean(normalized)), 1.0)

    def test_bernoulli_2x2_analytic(self):
        values, probabilities, mean_spacing, normalized = bernoulli_2x2_analytic()
        self.assertTrue(np.allclose(values, [0.0, 4.0, 4.0 * np.sqrt(2)]))
        self.assertTrue(np.allclose(probabilities, [0.25, 0.5, 0.25]))
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0)
        self.assertAlmostEqual(mean_spacing, 2.0 + np.sqrt(2))
        self.assertTrue(np.allclose(normalized, values / mean_spacing))


if __name__ == '__main__':
    unittest.main()
