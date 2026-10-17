from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

TARGET = "logP"
NAME = "Вещество"


def load_data(path=None):
    source = Path(path) if path else Path(__file__).resolve().parent / "data" / "lipo.csv"
    frame = pd.read_csv(source)
    if TARGET not in frame or NAME not in frame:
        raise ValueError("В таблице должны быть столбцы 'Вещество' и 'logP'.")
    return frame


class TwoLayerPerceptron:
    def __init__(self, hidden_size=8, learning_rate=0.01, tolerance=1e-6, max_iter=20000, decay=0.0, random_state=42):
        if hidden_size not in (2, 4, 8, 16, 32):
            raise ValueError("Число нейронов скрытого слоя должно быть 2, 4, 8, 16 или 32.")
        if learning_rate <= 0 or tolerance <= 0 or max_iter < 1 or decay < 0:
            raise ValueError("Параметры обучения заданы некорректно.")
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.decay = decay
        self.random_state = random_state
        self.x_scaler = StandardScaler()
        self.y_mean = None
        self.y_scale = None
        self.w1 = self.b1 = self.w2 = self.b2 = None
        self.n_iter_ = 0
        self.converged_ = False
        self.loss_ = None
        self.loss_history_ = []

    def _forward(self, x):
        hidden = np.maximum(x @ self.w1 + self.b1, 0.0)
        output = hidden @ self.w2 + self.b2
        return hidden, output

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if x.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError("X и y должны содержать одинаковое число наблюдений.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Данные должны быть конечными числами.")
        xs = self.x_scaler.fit_transform(x)
        self.y_mean = float(y.mean())
        self.y_scale = float(y.std()) or 1.0
        ys = (y - self.y_mean) / self.y_scale
        rng = np.random.default_rng(self.random_state)
        self.w1 = rng.normal(0, np.sqrt(2 / xs.shape[1]), (xs.shape[1], self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size))
        self.w2 = rng.normal(0, np.sqrt(2 / self.hidden_size), (self.hidden_size, 1))
        self.b2 = np.zeros((1, 1))
        params = [self.w1, self.b1, self.w2, self.b2]
        first = [np.zeros_like(p) for p in params]
        second = [np.zeros_like(p) for p in params]
        previous_loss = np.inf
        stale = 0
        for step in range(1, self.max_iter + 1):
            hidden, output = self._forward(xs)
            error = output - ys
            loss = float(np.mean(error ** 2))
            grad_output = 2 * error / len(xs)
            grads = [xs.T @ ((grad_output @ self.w2.T) * (hidden > 0)),
                     np.sum((grad_output @ self.w2.T) * (hidden > 0), axis=0, keepdims=True),
                     hidden.T @ grad_output,
                     np.sum(grad_output, axis=0, keepdims=True)]
            rate = self.learning_rate / (1 + self.decay * (step - 1))
            for i, (param, grad) in enumerate(zip(params, grads)):
                first[i] = 0.9 * first[i] + 0.1 * grad
                second[i] = 0.999 * second[i] + 0.001 * grad ** 2
                m_hat = first[i] / (1 - 0.9 ** step)
                v_hat = second[i] / (1 - 0.999 ** step)
                param -= rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            self.n_iter_ = step
            if step == 1 or step % 100 == 0:
                self.loss_history_.append(loss)
            if abs(previous_loss - loss) < self.tolerance:
                stale += 1
            else:
                stale = 0
            if stale >= 30:
                self.converged_ = True
                break
            previous_loss = loss
        _, final_output = self._forward(xs)
        self.loss_ = float(np.mean((final_output - ys) ** 2))
        return self

    def predict(self, x):
        if self.w1 is None:
            raise RuntimeError("Сначала нужно обучить модель методом fit().")
        xs = self.x_scaler.transform(np.asarray(x, dtype=float))
        _, output = self._forward(xs)
        return output.ravel() * self.y_scale + self.y_mean


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mse = float(np.mean((y_true - y_pred) ** 2))
    tss = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - np.sum((y_true - y_pred) ** 2) / tss) if tss else float("nan")
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "r2": r2}


def compare_models(frame=None, path=None, max_iter=20000):
    frame = frame if frame is not None else load_data(path)
    names = [c for c in frame.columns if c not in (NAME, TARGET)]
    x = frame[names].to_numpy(dtype=float)
    y = frame[TARGET].to_numpy(dtype=float)
    results = []
    models = {}
    for size in (2, 4, 8, 16, 32):
        model = TwoLayerPerceptron(size, learning_rate=0.01, tolerance=1e-6, max_iter=max_iter, decay=0.00005).fit(x, y)
        metrics = regression_metrics(y, model.predict(x))
        results.append({"Модель": f"Ручная, {size} нейронов", "Итерации": model.n_iter_, "Сошлась": model.converged_, **metrics})
        models[size] = model
    fixed_model = TwoLayerPerceptron(8, learning_rate=0.01, tolerance=1e-6, max_iter=max_iter, decay=0.0).fit(x, y)
    fixed_metrics = regression_metrics(y, fixed_model.predict(x))
    results.append({"Модель": "Ручная, 8 нейронов, постоянный шаг", "Итерации": fixed_model.n_iter_, "Сошлась": fixed_model.converged_, **fixed_metrics})
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x)
    scaled_y = (y - y.mean()) / (y.std() or 1)
    sklearn_model = MLPRegressor(hidden_layer_sizes=(8,), activation="relu", solver="adam", learning_rate_init=0.01, max_iter=max_iter, tol=1e-6, random_state=42)
    sklearn_model.fit(scaled_x, scaled_y)
    sklearn_prediction = sklearn_model.predict(scaled_x) * (y.std() or 1) + y.mean()
    results.append({"Модель": "scikit-learn, 8 нейронов", "Итерации": int(sklearn_model.n_iter_), "Сошлась": bool(sklearn_model.n_iter_ < max_iter), **regression_metrics(y, sklearn_prediction)})
    return pd.DataFrame(results), models, {"model": sklearn_model, "scaler": scaler}
