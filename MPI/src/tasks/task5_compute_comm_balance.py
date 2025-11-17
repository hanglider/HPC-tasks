from mpi4py import MPI
import numpy as np
import time

def fake_compute(work_iters: int):
    """
    Имитирует вычислительную нагрузку.
    Просто бессмысленный цикл.
    """
    x = 0
    for i in range(work_iters):
        x += i * 0.000001
    return x


def fake_communication(comm: MPI.Comm, msg_size: int, rounds: int):
    """
    Имитирует коммуникационную нагрузку.
    Каждый процесс отправляет сообщение следующему
    (кольцо ring topology).
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    send_to   = (rank + 1) % size
    recv_from = (rank - 1 + size) % size

    buf_send = np.ones(msg_size, dtype=np.byte)
    buf_recv = np.empty(msg_size, dtype=np.byte)

    for _ in range(rounds):
        req_s = comm.Isend(buf_send, dest=send_to)
        req_r = comm.Irecv(buf_recv, source=recv_from)
        MPI.Request.Waitall([req_s, req_r])


def run_task5_compute_comm_balance(
        compute_iters: int,
        msg_size: int,
        comm_rounds: int,
        comm: MPI.Comm = MPI.COMM_WORLD):
    """
    compute_iters  — сколько "вычислений" делать каждому процессу
    msg_size      — размер одного сообщения
    comm_rounds   — сколько коммуникационных обменов сделать
    """

    rank = comm.Get_rank()
    comm.Barrier()
    t0 = time.perf_counter()

    # 1) вычислительная часть
    _ = fake_compute(compute_iters)

    # 2) коммуникационная часть
    fake_communication(comm, msg_size, comm_rounds)

    comm.Barrier()
    t1 = time.perf_counter()

    total_time = t1 - t0

    # возвращаем время только с rank 0
    if rank == 0:
        return total_time
    else:
        return None