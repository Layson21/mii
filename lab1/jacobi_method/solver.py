import math
import numpy as np


def _validate_matrix(matrix, epsilon, max_iterations):
    if epsilon <= 0:
        raise ValueError("Точность должна быть положительной")
    if max_iterations <= 0:
        raise ValueError("Максимальное число итераций должно быть положительным")

    try:
        a = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("Элементы матрицы должны быть числами")

    if a.ndim != 2 or a.shape[0] == 0:
        raise ValueError("Матрица не должна быть пустой")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Матрица должна быть квадратной")
    if not np.all(np.isfinite(a)):
        raise ValueError("Элементы матрицы должны быть конечными числами")
    if not np.array_equal(a, a.T):
        raise ValueError("Матрица должна быть симметричной")

    return a.copy()


def jacobi_eigen(matrix, epsilon=1e-10, max_iterations=100000):
    a = _validate_matrix(matrix, epsilon, max_iterations)
    n = a.shape[0]
    v = np.eye(n)
    s = np.sum(np.triu(a, 1) ** 2)

    if n == 1:
        return np.diag(a), v

    p, q = 0, 1
    iteration = 0

    while s >= epsilon:
        if iteration >= max_iterations:
            raise RuntimeError("Метод Якоби не сошелся за заданное число итераций")

        apq = a[p, q]

        if apq != 0:
            c_value = (a[q, q] - a[p, p]) / (2 * apq)

            if c_value >= 0:
                tangent = 1 / (c_value + math.sqrt(c_value ** 2 + 1))
            else:
                tangent = 1 / (c_value - math.sqrt(c_value ** 2 + 1))

            cosine = 1 / math.sqrt(1 + tangent ** 2)
            sine = tangent * cosine

            a[p, p] = a[p, p] - apq * tangent
            a[q, q] = a[q, q] + apq * tangent
            a[p, q] = 0
            a[q, p] = 0

            old_a = a.copy()
            old_v = v.copy()

            for r in range(n):
                if r != p and r != q:
                    a[r, p] = old_a[r, p] - sine * (old_a[r, q] + sine / (1 + cosine) * old_a[r, p])
                    a[p, r] = a[r, p]
                    a[r, q] = old_a[r, q] + sine * (old_a[r, p] - sine / (1 + cosine) * old_a[r, q])
                    a[q, r] = a[r, q]
                v[r, p] = old_v[r, p] * cosine - old_v[r, q] * sine
                v[r, q] = old_v[r, p] * sine + old_v[r, q] * cosine

            s = s - apq ** 2

        iteration += 1
        q += 1

        if q >= n:
            p += 1
            if p >= n - 1:
                p = 0
            q = p + 1

    return np.diag(a), v