from mpi4py import MPI
import numpy as np
from typing import Tuple

def run_task1_minmax(vector_size: int, find_min: bool, comm: MPI.Comm = MPI.COMM_WORLD) -> float:
    """
    Возвращает глобальный min/max по вектору заданной длины.
    Вектор генерится на ранге 0, затем делится через Scatterv.
    :param vector_size: длина вектора
    :param find_min: True → ищем минимум, False → максимум
    :param comm: коммуникатор (по умолчанию MPI.COMM_WORLD)
    :return: глобальный результат (float)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Подготовим равномерные counts/offsets для Scatterv
    base = vector_size // size
    rem = vector_size % size
    counts = np.array([base + (1 if r < rem else 0) for r in range(size)], dtype=np.int32)
    displs = np.array([int(np.sum(counts[:r])) for r in range(size)], dtype=np.int32)

    # Данные только на ранге 0
    if rank == 0:
        full = np.random.uniform(-100.0, 100.0, vector_size).astype(np.float64)
    else:
        full = None

    # Буфер под локальный фрагмент
    local = np.empty(counts[rank], dtype=np.float64)

    # Распределяем фрагменты
    comm.Scatterv([full, counts, displs, MPI.DOUBLE], local, root=0)

    # Локальный результат
    local_res = np.min(local) if find_min else np.max(local)

    # Глобальная редукция
    op = MPI.MIN if find_min else MPI.MAX
    global_res = comm.allreduce(local_res, op=op)

    return float(global_res)