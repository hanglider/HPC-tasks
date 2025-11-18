"""
plot_all_results.py — универсальный визуализатор результатов OpenMP и MPI.
Обходит все подпапки проекта, ищет CSV-файлы в папках results/,
строит графики времени, ускорения и эффективности рядом с ними.

Теперь добавлена ПРАВИЛЬНАЯ поддержка MPI Задачи 6:
— task6_matrix_send_modes: строится график времени передачи данных
  для разных режимов send/ssend/rsend/bsend.
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


# ============================================================
#     РИСОВАНИЕ ГРАФИКА ДЛЯ ЗАДАЧИ 6  (send / ssend / rsend / bsend)
# ============================================================

def plot_task6_send_modes(df, csv_path):
    """
    Строит barplot:
         Время передачи данных vs режим передачи
    Размеры матриц (N) — разными цветами.
    """
    task_name = csv_path.stem

    if not {"mode", "size", "time"}.issubset(df.columns):
        print(f"⚠️ {csv_path}: нет колонок для рисования mode/size/time")
        return

    df_sorted = df.copy()
    df_sorted["mode"] = pd.Categorical(
        df_sorted["mode"],
        categories=["send", "ssend", "rsend", "bsend"],
        ordered=True,
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_sorted,
        x="mode",
        y="time",
        hue="size",
        palette="viridis"
    )

    plt.xlabel("Режим передачи данных (MPI mode)")
    plt.ylabel("Время передачи, сек")
    plt.title(f"Сравнение режимов передачи данных — {task_name}")
    plt.legend(title="Размер матрицы N")
    plt.tight_layout()

    out_file = csv_path.with_name(f"{csv_path.stem}_send_modes.png")
    plt.savefig(out_file, dpi=150)
    plt.close()

    print(f"📌 [{task_name}] barplot сохранён: {out_file.name}")



# ============================================================
#     ОБРАБОТКА ОДНОГО CSV-ФАЙЛА (для задач 1–5)
# ============================================================

def process_csv_default(df, csv_path):
    """Стандартные графики: time / speedup / efficiency (для задач 1–5)"""

    task_name = csv_path.stem
    df["task"] = df.get("task", task_name)

    # Проверяем наличие столбцов
    if not {"threads", "size", "time"}.issubset(df.columns):
        print(f"⚠️ {csv_path}: нет нужных колонок (threads, size, time)")
        return

    # Поиск baseline (threads=1)
    base_times = df[df["threads"] == 1].set_index("size")["time"]

    df["speedup"] = None
    df["efficiency"] = None

    # === Расчёт ускорения и эффективности ===
    for task in df["task"].unique():
        sub_task = df[df["task"] == task]

        for size in sub_task["size"].unique():
            sub = sub_task[sub_task["size"] == size].copy()
            base = sub[sub["threads"] == 1]["time"].min()

            # Если нет p=1 — пропускаем (speedup будет пустой)
            if pd.isna(base):
                continue

            df.loc[(df["task"] == task) & (df["size"] == size), "speedup"] = \
                base / sub["time"].values

            df.loc[(df["task"] == task) & (df["size"] == size), "efficiency"] = \
                (base / sub["time"].values) / sub["threads"].values

    # === Рисуем time / speedup / efficiency ===
    for metric, ylabel in [
        ("time", "Время выполнения, с"),
        ("speedup", "Ускорение S = T₁ / Tₚ"),
        ("efficiency", "Эффективность E = S / p")
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

    print(f"✅ {csv_path}: построены стандартные графики")


# ============================================================
#     ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def process_csv(csv_path: Path):
    """
    Определяет, для какой задачи строить графики:
      — task6_matrix_send_modes → ГРАФИК РЕЖИМОВ ПЕРЕДАЧИ
      — остальные → стандартные графики
    """

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️ {csv_path}: не удалось прочитать ({e})")
        return

    task_name = csv_path.stem

    # ======== КАСТОМНАЯ ЛОГИКА ДЛЯ ЗАДАЧИ 6 ========
    if "task6_matrix_send_modes" in task_name:
        plot_task6_send_modes(df, csv_path)
        return

    # ======== ДЛЯ ВСЕХ ОСТАЛЬНЫХ ========
    process_csv_default(df, csv_path)


# ============================================================

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