from mpi4py import MPI
import numpy as np
import time


def broadcast_B(B: np.ndarray, mode: str, comm: MPI.Comm):
    """
    Рассылка матрицы B от rank 0 ко всем процессам разными режимами.
    mode ∈ {"send", "ssend", "rsend", "bsend"}
    Возвращает время рассылки (сек).
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    tag = 0

    # --- обычный Send/Recv ---
    if mode == "send":
        comm.Barrier()
        t0 = time.perf_counter()
        if rank == 0:
            for dest in range(1, size):
                comm.Send(B, dest=dest, tag=tag)
        else:
            comm.Recv(B, source=0, tag=tag)
        comm.Barrier()
        t1 = time.perf_counter()
        return t1 - t0

    # --- синхронный Ssend/Recv ---
    if mode == "ssend":
        comm.Barrier()
        t0 = time.perf_counter()
        if rank == 0:
            for dest in range(1, size):
                comm.Ssend(B, dest=dest, tag=tag)
        else:
            comm.Recv(B, source=0, tag=tag)
        comm.Barrier()
        t1 = time.perf_counter()
        return t1 - t0

    # --- по готовности Rsend/Recv ---
    if mode == "rsend":
        # Все, кроме 0, заранее постят Irecv, чтобы гарантировать "готовность"
        req = None
        if rank != 0:
            req = comm.Irecv(B, source=0, tag=tag)

        comm.Barrier()
        t0 = time.perf_counter()

        if rank == 0:
            for dest in range(1, size):
                comm.Rsend(B, dest=dest, tag=tag)
        else:
            req.Wait()

        comm.Barrier()
        t1 = time.perf_counter()
        return t1 - t0

    if mode == "bsend":
    # размер буфера должен покрывать ВСЕ отправки
        if rank == 0:
            num_sends = size - 1
            buf_bytes = num_sends * (B.nbytes + MPI.BSEND_OVERHEAD)
            buf = np.empty(buf_bytes, dtype=np.byte)
            MPI.Attach_buffer(buf)

        comm.Barrier()
        t0 = time.perf_counter()

        if rank == 0:
            for dest in range(1, size):
                comm.Bsend(B, dest=dest, tag=tag)
        else:
            comm.Recv(B, source=0, tag=tag)

        comm.Barrier()
        t1 = time.perf_counter()

        if rank == 0:
            MPI.Detach_buffer()

        return t1 - t0


    raise ValueError(f"Unknown mode: {mode}")


def run_task6_matrix_send_modes(n: int,
                                mode: str,
                                comm: MPI.Comm = MPI.COMM_WORLD):
    """
    Умножение матриц C = A * B с разными режимами рассылки B.
    - n: размер матриц n x n
    - mode: "send" | "ssend" | "rsend" | "bsend"

    Возвращает время ТОЛЬКО передачи B (сек) на rank 0, иначе None.
    """

    rank = comm.Get_rank()
    size = comm.Get_size()

    if n % size != 0:
        if rank == 0:
            print(f"❌ N={n} не делится на число процессов {size}")
        return None

    block = n // size

    # --- инициализация матриц ---
    if rank == 0:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
    else:
        A = None
        B = np.empty((n, n), dtype=np.float64)

    # Рассылаем B выбранным режимом и меряем время
    comm_time = broadcast_B(B, mode, comm)

    # --- раздаём строки A обычным способом (это не часть эксперимента) ---
    A_local = np.empty((block, n), dtype=np.float64)
    if rank == 0:
        for r in range(size):
            s = r * block
            e = (r + 1) * block
            if r == 0:
                A_local[:, :] = A[s:e, :]
            else:
                comm.Send(A[s:e, :], dest=r, tag=1)
    else:
        comm.Recv(A_local, source=0, tag=1)

    # --- локальное умножение (чтобы программа была "реальной") ---
    C_local = A_local @ B

    # Собираем C на rank 0 (просто чтобы не ругался оптимизатор, время не меряем)
    if rank == 0:
        C = np.empty((n, n), dtype=np.float64)
        C[0:block, :] = C_local
        for r in range(1, size):
            s = r * block
            e = (r + 1) * block
            comm.Recv(C[s:e, :], source=r, tag=2)
    else:
        comm.Send(C_local, dest=0, tag=2)

    # Возвращаем именно время коммуникации
    return comm_time if rank == 0 else None