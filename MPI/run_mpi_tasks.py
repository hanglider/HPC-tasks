#!/usr/bin/env python3
import json
import shutil
import subprocess
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"

def which_mpi() -> str:
    for exe in ("mpirun", "mpiexec"):
        path = shutil.which(exe)
        if path:
            return path
    print("❌ Не найден mpirun/mpiexec. Установи Open MPI: brew install open-mpi")
    sys.exit(1)

def load_config():
    if not CONFIG_PATH.exists():
        print("❌ Не найден MPI/config.json")
        sys.exit(1)

    cfg = json.loads(CONFIG_PATH.read_text())

    script = cfg.get("script", "src/main.py")

    # 1) если есть np — используем его
    np_values = cfg.get("np")
    # 2) если np нет — используем processes
    if np_values is None:
        np_values = cfg.get("processes", [1, 2, 4, 8])

    # привести к списку
    if isinstance(np_values, int):
        np_values = [np_values]

    repeats = int(cfg.get("repeats", 1))
    extra_args = cfg.get("extra_args", [])
    if not isinstance(extra_args, list):
        print("⚠️ 'extra_args' в config.json должен быть списком строк. Игнорирую.")
        extra_args = []

    return script, np_values, repeats, extra_args

def main():
    mpirunner = which_mpi()
    script, np_values, repeats, extra_args = load_config()
    script_path = BASE_DIR / script
    if not script_path.exists():
        print(f"❌ Скрипт не найден: {script_path}")
        sys.exit(1)

    print(f"▶️  Скрипт: {script_path}")
    print(f"🧩 Потоки: {np_values}, Повторы: {repeats}")
    if extra_args:
        print(f"⚙️  Доп. аргументы mpirun: {' '.join(extra_args)}")

    for np_ in np_values:
        print(f"\n🚀 mpirun -n {np_} python3 {script}")
        for r in range(1, repeats + 1):
            PYTHON_PATH = "/opt/homebrew/bin/python3"
            cmd = [mpirunner, "-n", str(np_)] + extra_args + [PYTHON_PATH, str(script_path)]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Ошибка запуска (np={np_}, run={r}): {e}")
                sys.exit(e.returncode)

    print("\n🎉 Готово: все запуски завершены. Результаты смотри в MPI/results/")

if __name__ == "__main__":
    main()