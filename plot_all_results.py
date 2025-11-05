"""
plot_all_results.py — универсальный визуализатор результатов OpenMP и MPI.
Обходит все подпапки проекта, ищет CSV-файлы в папках results/,
строит графики времени, ускорения и эффективности рядом с ними.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style="whitegrid")
plt.rcParams.update({
    "figure.figsize": (7, 5),
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10
})

def process_csv(csv_path: Path):
    """Рисует графики для одного CSV"""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️ {csv_path}: не удалось прочитать ({e})")
        return

    if not {"threads", "size", "time"}.issubset(df.columns):
        print(f"⚠️ {csv_path}: нет нужных колонок (threads, size, time)")
        return

    task_name = csv_path.stem
    df["task"] = df.get("task", task_name)
    base_times = df[df["threads"] == 1].set_index("size")["time"]

    # === Расчёт ускорения и эффективности ===
    df["speedup"] = None
    df["efficiency"] = None

    for task in df["task"].unique():
        sub_task = df[df["task"] == task]
        for size in sub_task["size"].unique():
            sub = sub_task[sub_task["size"] == size].copy()
            base = sub[sub["threads"] == 1]["time"].min()
            df.loc[(df["task"] == task) & (df["size"] == size), "speedup"] = base / sub["time"].values
            df.loc[(df["task"] == task) & (df["size"] == size), "efficiency"] = (base / sub["time"].values) / sub["threads"].values


    # === Графики ===
    for metric, ylabel, formula in [
        ("time", "Время выполнения, с", None),
        ("speedup", "Ускорение S = T₁ / Tₚ", "speedup"),
        ("efficiency", "Эффективность E = S / p", "efficiency")
    ]:
        plt.figure()
        for size in sorted(df["size"].unique()):
            sub = df[df["size"] == size]
            plt.plot(sub["threads"], sub[metric], marker="o", label=f"N={size}")
        plt.xlabel("Число потоков / процессов")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} — {task_name}")
        plt.legend()
        plt.tight_layout()
        out_file = csv_path.with_name(f"{csv_path.stem}_{metric}.png")
        plt.savefig(out_file, dpi=150)
        plt.close()

    print(f"✅ {csv_path}: построены графики")

def main():
    root = Path(__file__).parent
    all_csv = list(root.rglob("results/*.csv"))

    if not all_csv:
        print("⚠️ Не найдено ни одного CSV-файла в подпапках results/")
        return

    print(f"📂 Найдено {len(all_csv)} CSV-файлов:")
    for f in all_csv:
        print(f"   {f.relative_to(root)}")
        process_csv(f)

    print("🎉 Все графики сохранены рядом с исходными CSV.")

if __name__ == "__main__":
    main()