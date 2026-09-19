import numpy as np
import pytest

from jacobi_method import jacobi_eigen


def test_diagonal_matrix():
    matrix = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    eigenvalues, eigenvectors = jacobi_eigen(matrix)
    assert np.allclose(eigenvalues, [2.0, 5.0, -1.0])
    assert np.allclose(eigenvectors, np.eye(3))


def test_two_by_two_matrix():
    matrix = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
    ])
    eigenvalues, eigenvectors = jacobi_eigen(matrix)
    assert np.allclose(np.sort(eigenvalues), [1.0, 3.0])
    assert np.allclose(matrix @ eigenvectors, eigenvectors @ np.diag(eigenvalues))


def test_three_by_three_matrix():
    matrix = np.array([
        [4.0, 1.0, 1.0],
        [1.0, 3.0, 0.0],
        [1.0, 0.0, 2.0],
    ])
    eigenvalues, eigenvectors = jacobi_eigen(matrix, epsilon=1e-14)
    expected = [1.467911113762044, 2.652703644666139, 4.879385241571817]
    assert np.allclose(np.sort(eigenvalues), expected)
    assert np.allclose(matrix @ eigenvectors, eigenvectors @ np.diag(eigenvalues))


def test_eigenvectors_are_orthonormal():
    matrix = np.array([
        [5.0, 2.0, 1.0],
        [2.0, 3.0, 1.0],
        [1.0, 1.0, 2.0],
    ])
    _, eigenvectors = jacobi_eigen(matrix, epsilon=1e-14)
    assert np.allclose(eigenvectors.T @ eigenvectors, np.eye(3))


def test_one_by_one_matrix():
    eigenvalues, eigenvectors = jacobi_eigen([[7.0]])
    assert np.allclose(eigenvalues, [7.0])
    assert np.allclose(eigenvectors, [[1.0]])


def test_non_square_matrix():
    with pytest.raises(ValueError, match="квадратной"):
        jacobi_eigen([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0]])


def test_non_symmetric_matrix():
    with pytest.raises(ValueError, match="симметричной"):
        jacobi_eigen([[1.0, 2.0], [3.0, 4.0]])


def test_empty_matrix():
    with pytest.raises(ValueError, match="пустой"):
        jacobi_eigen([])


def test_invalid_element_type():
    with pytest.raises(TypeError, match="числами"):
        jacobi_eigen([[1.0, "x"], ["x", 2.0]])


def test_invalid_epsilon():
    with pytest.raises(ValueError, match="Точность"):
        jacobi_eigen([[1.0]], epsilon=0)


def test_iteration_limit():
    matrix = np.array([
        [4.0, 1.0, 1.0],
        [1.0, 3.0, 1.0],
        [1.0, 1.0, 2.0],
    ])
    with pytest.raises(RuntimeError, match="не сошелся"):
        jacobi_eigen(matrix, epsilon=1e-30, max_iterations=1)
