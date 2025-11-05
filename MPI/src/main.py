import json, time
from pathlib import Path
import pandas as pd
from mpi4py import MPI
from tasks import TASKS

def measure_min_time(fn, repeats: int, *args, **kwargs) -> float:
    comm = MPI.COMM_WORLD
    best = float("inf")
    for _ in range(repeats):
        comm.Barrier()                # синхронизация перед замером
        t0 = time.perf_counter()
        _ = fn(*args, **kwargs)
        comm.Barrier()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best

def main():
    base_dir = Path(__file__).parent.parent
    cfg_path = base_dir / "config.json"
    if not cfg_path.exists():
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("❌ Не найден MPI/config.json")
        return

    cfg = json.loads(cfg_path.read_text())
    task_name = cfg.get("task", "task1_minmax")
    sizes     = cfg.get("sizes", [10_000_000])
    repeats   = cfg.get("repeats", 3)
    find_min  = cfg.get("find_min", True)
    out_path  = base_dir / "results" / f"{task_name}.csv"
    out_path.parent.mkdir(exist_ok=True)

    if task_name not in TASKS:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"❌ Неизвестная задача: {task_name}")
        return

    fn = TASKS[task_name]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    procs = comm.Get_size()

    for n in sizes:
        # 1 прогон для получения значения
        value = fn(n, find_min, comm=comm)
        # бенчмарк
        tsec = measure_min_time(fn, repeats, n, find_min, comm=comm)

        if rank == 0:
            row = pd.DataFrame([{
                "task": task_name,
                "threads": procs,     # совместимо с визуализатором
                "size": n,
                "time": tsec,
                "result": value
            }])
            if out_path.exists():
                row.to_csv(out_path, mode="a", header=False, index=False)
            else:
                row.to_csv(out_path, index=False)
            print(f"[{procs} потоков] {task_name}, N={n}, time={tsec:.6f}s, result={value:.5f}")

if __name__ == "__main__":
    main()