import json
import time
import inspect
from pathlib import Path
import pandas as pd
from mpi4py import MPI

from tasks.task5_compute_comm_balance import run_task5_compute_comm_balance


def measure_min_time(fn, repeats: int, *args, **kwargs) -> float:
    """Замеряет лучшее время выполнения среди repeats запусков."""
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
    sizes        = cfg.get("sizes", [100000])
    compute_it   = cfg.get("compute_iters", 5000000)
    msg_size     = cfg.get("msg_size", 100000)
    comm_rounds  = cfg.get("comm_rounds", 20)
    repeats      = cfg.get("repeats", 3)

    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    out_path = results_dir / "task5_compute_comm_balance.csv"

    # ---- запуск только задачи 5 ----
    for n in sizes:
        args = [compute_it, msg_size, comm_rounds]
        kwargs = {"comm": comm}

        # Один раз получаем значение (только rank 0)
        value = run_task5_compute_comm_balance(*args, **kwargs)

        # Измеряем лучшее время
        tsec = measure_min_time(run_task5_compute_comm_balance,
                                repeats,
                                *args,
                                **kwargs)

        if rank == 0:
            row = pd.DataFrame([{
                "task": "task5_compute_comm_balance",
                "threads": procs,
                "size": n,
                "compute_iters": compute_it,
                "msg_size": msg_size,
                "comm_rounds": comm_rounds,
                "time": tsec,
                "result": value
            }])

            if out_path.exists():
                row.to_csv(out_path, mode="a", header=False, index=False)
            else:
                row.to_csv(out_path, index=False)

            print(f"[np={procs}] N={n}, time={tsec:.6f}s, result={value}")


if __name__ == "__main__":
    main()