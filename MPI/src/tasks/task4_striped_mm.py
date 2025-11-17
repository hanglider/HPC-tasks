from mpi4py import MPI
import numpy as np

def run_task4_striped_mm(n: int, comm: MPI.Comm = MPI.COMM_WORLD):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Размер матриц: n x n
    # На root генерируем данные
    if rank == 0:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
    else:
        A = None
        B = None

    # Раздаем B всем процессам
    B = comm.bcast(B, root=0)

    # Делим строки A поровну
    rows_per_proc = n // size
    extra = n % size

    # Каждому процессу своё количество строк
    if rank < extra:
        start = rank * (rows_per_proc + 1)
        end = start + rows_per_proc + 1
    else:
        start = rank * rows_per_proc + extra
        end = start + rows_per_proc

    local_rows = end - start

    # Буфер для подматрицы
    A_local = np.zeros((local_rows, n))

    # root отправляет строки
    if rank == 0:
        for r in range(size):
            if r == 0:
                A_local[:, :] = A[start:end, :]
                continue
            
            if r < extra:
                s = r * (rows_per_proc + 1)
                e = s + rows_per_proc + 1
            else:
                s = r * rows_per_proc + extra
                e = s + rows_per_proc

            comm.send(A[s:e, :], dest=r)
    else:
        A_local = comm.recv(source=0)

    # LOCALLY compute product
    C_local = A_local @ B

    # Gather sizes
    sizes = comm.gather(C_local.shape[0], root=0)

    # Root собирает C
    if rank == 0:
        C = np.zeros((n, n))
        # вставляем свой кусок
        C[start:end, :] = C_local

        recv_offset = 0
        # начинаем вставлять после собственного куска
        for r in range(1, size):
            recv_rows = sizes[r]

            # вычисляем его индекс
            if r < extra:
                s = r * (rows_per_proc + 1)
                e = s + rows_per_proc + 1
            else:
                s = r * rows_per_proc + extra
                e = s + rows_per_proc

            block = comm.recv(source=r)
            C[s:e, :] = block

        # Возвращаем реультат (root)
        return C
    else:
        comm.send(C_local, dest=0)
        return None