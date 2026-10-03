import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MATRIX_SIZES = (2, 4, 16)
ENSEMBLE_SIZE = 10000
SEED = 42
OUTPUT_DIR = Path('results')

__all__ = [
    'wigner_distribution',
    'generate_matrix',
    'eigenvalue_spacing',
    'generate_spacings',
    'normalize_spacings',
    'histogram_error',
    'bernoulli_2x2_analytic'
]


def wigner_distribution(s):
    return np.pi * s / 2 * np.exp(-np.pi * s**2 / 4)


def generate_matrix(size, distribution, rng):
    if distribution == 'normal':
        matrix = rng.normal(0.0, 1.0, (size, size))
    elif distribution == 'bernoulli':
        matrix = rng.choice((-1.0, 1.0), size=(size, size))
    else:
        raise ValueError('Неизвестное распределение')
    return matrix + matrix.T


def eigenvalue_spacing(matrix):
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues.sort()
    index = len(eigenvalues) // 2 - 1
    return eigenvalues[index + 1] - eigenvalues[index]


def generate_spacings(size, distribution, ensemble_size, rng):
    spacings = np.empty(ensemble_size)
    for i in range(ensemble_size):
        matrix = generate_matrix(size, distribution, rng)
        spacings[i] = eigenvalue_spacing(matrix)
    return spacings


def normalize_spacings(spacings):
    mean_spacing = np.mean(spacings)
    if mean_spacing == 0:
        return spacings.copy(), mean_spacing
    return spacings / mean_spacing, mean_spacing


def plot_distribution(spacings, size, distribution, output_dir):
    x = np.linspace(0, max(4.0, float(np.max(spacings)) * 1.05), 1000)
    title_distribution = 'Нормальное распределение' if distribution == 'normal' else 'Значения -1 и 1'

    plt.figure(figsize=(9, 6))
    plt.hist(spacings, bins=60, density=True, alpha=0.65, label='Гистограмма')
    plt.plot(x, wigner_distribution(x), linewidth=2, label='Распределение Вигнера')
    plt.xlabel('Нормированное расстояние s')
    plt.ylabel('Плотность вероятности')
    plt.title(f'{title_distribution}, матрицы {size}x{size}')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    filename = output_dir / f'{distribution}_{size}x{size}.png'
    plt.savefig(filename, dpi=160)
    plt.close()
    return filename


def histogram_error(spacings):
    hist, edges = np.histogram(spacings, bins=80, range=(0, 4), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    reference = wigner_distribution(centers)
    return float(np.mean(np.abs(hist - reference)))


def bernoulli_2x2_analytic():
    mean_spacing = 2 + np.sqrt(2)
    values = np.array([0.0, 4.0, 4.0 * np.sqrt(2)])
    probabilities = np.array([0.25, 0.5, 0.25])
    normalized = values / mean_spacing
    return values, probabilities, mean_spacing, normalized


def save_summary(rows, output_dir):
    lines = [
        'distribution,size,mean_spacing_before_normalization,normalized_mean,mean_absolute_histogram_error'
    ]
    for row in rows:
        lines.append(','.join(map(str, row)))
    (output_dir / 'summary.csv').write_text('\n'.join(lines), encoding='utf-8')


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []

    for distribution in ('normal', 'bernoulli'):
        for size in MATRIX_SIZES:
            spacings = generate_spacings(size, distribution, ENSEMBLE_SIZE, rng)
            normalized, mean_spacing = normalize_spacings(spacings)
            error = histogram_error(normalized)
            plot_distribution(normalized, size, distribution, OUTPUT_DIR)
            rows.append((distribution, size, mean_spacing, float(np.mean(normalized)), error))
            print(
                f'{distribution:10s} {size:2d}x{size:<2d} '
                f'mean={mean_spacing:.6f} normalized_mean={np.mean(normalized):.6f} '
                f'error={error:.6f}'
            )

    save_summary(rows, OUTPUT_DIR)

    values, probabilities, mean_spacing, normalized = bernoulli_2x2_analytic()
    print('\nАналитический результат для 2x2, элементы -1 и 1:')
    for value, probability, norm_value in zip(values, probabilities, normalized):
        print(f's={value:.6f}, P={probability:.2f}, s_norm={norm_value:.6f}')
    print(f'Среднее расстояние: {mean_spacing:.6f}')


if __name__ == '__main__':
    main()
