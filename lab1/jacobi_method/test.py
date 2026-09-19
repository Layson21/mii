from solver import jacobi_eigen
import numpy as np


matrix = np.array([
    [2.0, 0.0, 0.0],
    [0.0, 5.0, 0.0],
    [0.0, 0.0, -1.0],
])
eigenvalues, eigenvectors = jacobi_eigen(matrix)
print(f"1){eigenvalues}\n{eigenvectors}\n")


matrix = np.array([
    [2.0, 1.0],
    [1.0, 2.0],
])
eigenvalues, eigenvectors = jacobi_eigen(matrix)
print(f"2){eigenvalues}\n{eigenvectors}\n")


matrix = np.array([
    [4.0, 1.0, 1.0],
    [1.0, 3.0, 0.0],
    [1.0, 0.0, 2.0],
])
eigenvalues, eigenvectors = jacobi_eigen(matrix, epsilon=1e-14)


matrix = np.array([
    [5.0, 2.0, 1.0],
    [2.0, 3.0, 1.0],
    [1.0, 1.0, 2.0],
])
eigenvalues, eigenvectors = jacobi_eigen(matrix, epsilon=1e-14)
print(f"4){eigenvectors.T @ eigenvectors}\n")


eigenvalues, eigenvectors = jacobi_eigen([[7.0]])
print(f"5){eigenvalues}\n{eigenvectors}\n")

