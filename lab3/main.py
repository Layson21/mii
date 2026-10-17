import argparse
from pathlib import Path
from model import load_data, fit_ols
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Лабораторная работа 3: линейная регрессия")
    parser.add_argument("--data", help="Путь к lipo.csv")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "results"),
        help="Каталог для результатов"
    )
    args = parser.parse_args()
    frame = load_data(args.data)
    result = fit_ols(frame)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result["coefficient_table"].to_csv(output / "coefficients.csv", index=False)

    pd.DataFrame({"Вещество": frame["Вещество"], "Факт": result["target"], "Прогноз": result["fitted"], "Остаток": result["residuals"]}).to_csv(output / "residuals.csv", index=False)
    print(f"Наблюдений: {len(frame)}")
    print(f"Дисперсия ошибки s²: {result['variance']:.6f}")
    print(f"Стандартная ошибка регрессии s: {result['standard_error']:.6f}")
    print(f"R²: {result['r_squared']:.6f}")
    print(f"Результаты сохранены в {output.resolve()}")


if __name__ == "__main__":
    main()
