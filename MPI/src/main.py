import json
import time
from pathlib import Path
import pandas as pd
from mpi4py import MPI

from tasks.task6_matrix_send_modes import run_task6_matrix_send_modes


def measure_min_time(fn, repeats: int, *args, **kwargs) -> float:
    """Замеряет лучшее время выполнения среди repeats."""
    comm = MPI.COMM_WORLD
    best = float("inf")

    for _ in range(repeats):
        comm.Barrier()
        t0 = time.perf_counter()
        _ = fn(*args, **kwargs)
        comm.Barrier()
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)

    return best


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    procs = comm.Get_size()

    base_dir = Path(__file__).parent.parent
    cfg_path = base_dir / "config.json"

    if not cfg_path.exists():
        if rank == 0:
            print("❌ Не найден MPI/config.json")
        return

    cfg = json.loads(cfg_path.read_text())

    # ---- читаем конфиг ----
    sizes   = cfg.get("sizes", [600])
    modes   = cfg.get("modes", ["send", "ssend", "rsend", "bsend"])
    repeats = cfg.get("repeats", 3)

    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    out_path = results_dir / "task6_matrix_send_modes.csv"

    # ---- запуск только задачи 6 ----
    for n in sizes:
        for mode in modes:

            args = [n, mode]
            kwargs = {"comm": comm}

            # 1) Один раз запускаем и получаем значение (rank 0)
            value = run_task6_matrix_send_modes(*args, **kwargs)

            # 2) Замеряем лучшее время
            tsec = measure_min_time(run_task6_matrix_send_modes,
                                    repeats,
                                    *args,
                                    **kwargs)

            # 3) Сохраняем CSV
            if rank == 0:
                row = pd.DataFrame([{
                    "task": "task6_matrix_send_modes",
                    "mode": mode,
                    "threads": procs,
                    "size": n,
                    "time": tsec,
                    "result": value
                }])

                if out_path.exists():
                    row.to_csv(out_path, mode="a", header=False, index=False)
                else:
                    row.to_csv(out_path, index=False)

                print(f"[np={procs}] mode={mode}, N={n}, time={tsec:.6f}s, result={value}")


if __name__ == "__main__":
    main()