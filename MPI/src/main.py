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

    # Если в конфиге указано требуемое число процессов — проверим
    required_procs = cfg.get("processes")
    if required_procs is not None:
        if isinstance(required_procs, int):
            required_procs = [required_procs]
        if procs not in required_procs:
            if rank == 0:
                print(f"❌ Этот запуск требует процессов {required_procs}, а запущено {procs}")
            return

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

            # Первый вызов — получаем результат (только rank 0 обычно что-то возвращает)
            value = fn(*args, **kwargs)

            # Измеряем минимальное время
            tsec = measure_min_time(fn, repeats, *args, **kwargs)

            if rank == 0:
                


                # спец-логика для ping-pong (task3)
                if task_name == "task3_pingpong":
                    # ВОТ ТАК ПРАВИЛЬНО:
                    # time = latency (из value: avg_time)
                    # result не нужен → ставим пусто
                    rec = {
                        "task": task_name,
                        "threads": procs,
                        "size": n,
                        "time": value,    # <- самое важное
                        "result": ""      # просто пусто
                    }
                else:
                    rec = {
                        "task": task_name,
                        "threads": procs,
                        "size": n,
                        "time": tsec,
                        "result": value
                    }

                row = pd.DataFrame([rec])




                if out_path.exists():
                    row.to_csv(out_path, mode="a", header=False, index=False)
                else:
                    row.to_csv(out_path, index=False)

                print(f"[{procs} процессов] {task_name}, N={n}, time={tsec:.6f}s, result={value:.5f}")

if __name__ == "__main__":
    main()