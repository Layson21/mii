from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

TARGET = "logP"
NAME = "Вещество"


def load_data(path=None):
    source = Path(path) if path else Path(__file__).resolve().parent / "data" / "lipo.csv"
    frame = pd.read_csv(source)
    if TARGET not in frame or NAME not in frame:
        raise ValueError("В таблице должны быть столбцы 'Вещество' и 'logP'.")
    return frame


def fit_ols(frame):
    feature_names = [c for c in frame.columns if c not in (NAME, TARGET)]
    x = frame[feature_names].to_numpy(dtype=float)
    y = frame[TARGET].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Матрица признаков имеет неполный ранг.")
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ beta
    residuals = y - fitted
    n, p = design.shape
    df_resid = n - p
    if df_resid <= 0:
        raise ValueError("Недостаточно наблюдений для оценки дисперсии.")
    sse = float(residuals @ residuals)
    variance = sse / df_resid
    standard_error = float(np.sqrt(variance))
    tss = float(((y - y.mean()) ** 2).sum())
    r_squared = float(1 - sse / tss) if tss else float("nan")
    covariance = variance * np.linalg.inv(design.T @ design)
    standard_errors = np.sqrt(np.diag(covariance))
    critical = t.ppf(0.975, df_resid)
    intervals = np.column_stack([beta - critical * standard_errors, beta + critical * standard_errors])
    coefficient_table = pd.DataFrame({
        "Регрессор": ["const"] + feature_names,
        "Коэффициент": beta,
        "Стандартная ошибка": standard_errors,
        "Нижняя граница 95%": intervals[:, 0],
        "Верхняя граница 95%": intervals[:, 1],
        "t-статистика": beta / standard_errors,
    })
    return {
        "feature_names": feature_names,
        "coefficients": beta,
        "fitted": fitted,
        "residuals": residuals,
        "variance": float(variance),
        "standard_error": standard_error,
        "r_squared": r_squared,
        "degrees_of_freedom": int(df_resid),
        "coefficient_table": coefficient_table,
        "design": design,
        "target": y,
    }


def predict(frame, coefficients):
    feature_names = [c for c in frame.columns if c not in (NAME, TARGET)]
    values = frame[feature_names].to_numpy(dtype=float)
    return np.column_stack([np.ones(len(values)), values]) @ np.asarray(coefficients, dtype=float)


def run_analysis(path=None):
    frame = load_data(path)
    result = fit_ols(frame)
    table = result["coefficient_table"].copy()
    table["Коэффициент"] = table["Коэффициент"].round(6)
    table["Стандартная ошибка"] = table["Стандартная ошибка"].round(6)
    table["Нижняя граница 95%"] = table["Нижняя граница 95%"].round(6)
    table["Верхняя граница 95%"] = table["Верхняя граница 95%"].round(6)
    table["t-статистика"] = table["t-статистика"].round(4)
    return result, table
