import json, time, inspect
from pathlib import Path
import pandas as pd
from mpi4py import MPI
from tasks import TASKS

def measure_min_time(fn, repeats: int, *args, **kwargs) -> float:
    comm = MPI.COMM_WORLD
    best = float("inf")
    for _ in range(repeats):
        comm.Barrier()
        t0 = time.perf_counter()
        _ = fn(*args, **kwargs)
        comm.Barrier()
        best = min(best, time.perf_counter() - t0)
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

    # ---- ЯВНО: либо "task", либо "tasks"
    tasks_field = cfg.get("tasks")
    task_field  = cfg.get("task")
    if tasks_field and task_field:
        if rank == 0:
            print("❌ Укажи либо 'task', либо 'tasks' (не оба).")
        return
    if tasks_field:
        task_names = list(tasks_field)
    elif task_field:
        task_names = [task_field]
    else:
        if rank == 0:
            print("❌ В config.json нужно указать 'task' или 'tasks'.")
        return
    # ----

    sizes    = cfg.get("sizes", [10_000_000])
    repeats  = cfg.get("repeats", 3)
    find_min = cfg.get("find_min", True)  # используется только теми задачами, кто его принимает

    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    for task_name in task_names:
        if task_name not in TASKS:
            if rank == 0:
                print(f"❌ Неизвестная задача: {task_name}. Доступные: {', '.join(TASKS.keys())}")
            continue

        fn = TASKS[task_name]
        fn_params = inspect.signature(fn).parameters
        out_path = results_dir / f"{task_name}.csv"

        for n in sizes:
            # Формируем аргументы динамически под сигнатуру функции
            args, kwargs = [], {"comm": comm}
            if "vector_size" in fn_params or "n" in fn_params:
                args.append(n)
            if "find_min" in fn_params:
                args.append(find_min)

            value = fn(*args, **kwargs)
            tsec  = measure_min_time(fn, repeats, *args, **kwargs)

            if rank == 0:
                row = pd.DataFrame([{
                    "task": task_name,
                    "threads": procs,
                    "size": n,
                    "time": tsec,
                    "result": value
                }])
                if out_path.exists():
                    row.to_csv(out_path, mode="a", header=False, index=False)
                else:
                    row.to_csv(out_path, index=False)

                print(f"[{procs} процессов] {task_name}, N={n}, time={tsec:.6f}s, result={value:.5f}")

if __name__ == "__main__":
    main()