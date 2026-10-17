import argparse
from pathlib import Path
from model import load_data, compare_models
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Лабораторная работа 4: двухслойный перцептрон")
    parser.add_argument("--data", help="Путь к lipo.csv")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "results"),
        help="Каталог для результатов"
    )
    parser.add_argument("--max-iter", type=int, default=20000)
    args = parser.parse_args()
    frame = load_data(args.data)
    table, models, sklearn_model = compare_models(frame, max_iter=args.max_iter)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "model_comparison.csv", index=False)
    x = frame[[c for c in frame.columns if c not in ("Вещество", "logP")]].to_numpy(dtype=float)
    for size, model in models.items():
        pd.DataFrame({"Вещество": frame["Вещество"], "Факт": frame["logP"], "Прогноз": model.predict(x), "Остаток": frame["logP"] - model.predict(x)}).to_csv(output / f"predictions_{size}.csv", index=False)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Результаты сохранены в {output.resolve()}")


if __name__ == "__main__":
    main()
