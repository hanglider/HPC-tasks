from mpi4py import MPI
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def run_task4_fox_mm(n: int, comm: MPI.Comm = MPI.COMM_WORLD):
    """
    Алгоритм Фокса для C = A * B.
    Требования:
      - число процессов size = p^2 (p = sqrt(size) целый)
      - n % p == 0
    Возвращает:
      - на rank 0: матрицу C (numpy.ndarray n x n)
      - на остальных рангах: None
    """

    rank = comm.Get_rank()
    size = comm.Get_size()

    # ---- проверки размера коммуникатора ----
    p = int(math.isqrt(size))
    if p * p != size:
        if rank == 0:
            print(f"❌ Fox requires p*p processes (4,9,16,...). Got {size}.")
        return None

    # блоки должны быть целого размера
    if n % p != 0:
        if rank == 0:
            print(f"❌ Matrix size n={n} must be divisible by sqrt(np)={p} for Fox.")
        return None

    block = n // p

    # Координаты процесса в решётке p x p
    row = rank // p
    col = rank % p

    # Создаем коммуникаторы строк и столбцов 2D-сетки
    # процессы с одинаковым row попадают в один row_comm
    row_comm = comm.Split(color=row, key=col)
    # процессы с одинаковым col попадают в один col_comm
    col_comm = comm.Split(color=col, key=row)

    # ---- инициализация матриц A и B на rank 0 ----
    if rank == 0:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
    else:
        A = None
        B = None

    # ---- раздача блоков A и B ----
    A_block = np.empty((block, block), dtype=np.float64)
    B_block = np.empty((block, block), dtype=np.float64)

    if rank == 0:
        for r in range(size):
            i = r // p
            j = r % p
            Ab = A[i*block:(i+1)*block, j*block:(j+1)*block]
            Bb = B[i*block:(i+1)*block, j*block:(j+1)*block]
            if r == 0:
                A_block[:, :] = Ab
                B_block[:, :] = Bb
            else:
                comm.send((Ab, Bb), dest=r)
    else:
        Ab, Bb = comm.recv(source=0)
        A_block[:, :] = Ab
        B_block[:, :] = Bb

    # ---- основной цикл Фокса ----
    C_block = np.zeros((block, block), dtype=np.float64)

    for k in range(p):
        # Выбор "ведущего" блока A в данной строке
        if col == k:
            pivot_block = A_block.copy()
        else:
            pivot_block = np.empty_like(A_block)

        # Рассылка pivot_block по строке (row_comm)
        # В row_comm ранги — это просто col (0..p-1)
        row_comm.Bcast(pivot_block, root=k)

        # Локальное умножение
        C_block += pivot_block @ B_block

        # Циклический сдвиг B вверх в столбце
        B_next = np.empty_like(B_block)
        send_to   = (row - 1) % p   # вверх по колонке
        recv_from = (row + 1) % p   # снизу

        col_comm.Sendrecv(sendbuf=B_block,
                          dest=send_to,   sendtag=0,
                          recvbuf=B_next,
                          source=recv_from, recvtag=0)

        B_block[:, :] = B_next

    # ---- сбор результата C на rank 0 ----
    if rank == 0:
        C = np.zeros((n, n), dtype=np.float64)
        # свой блок
        C[row*block:(row+1)*block, col*block:(col+1)*block] = C_block

        for r in range(1, size):
            cb = comm.recv(source=r)
            i = r // p
            j = r % p
            C[i*block:(i+1)*block, j*block:(j+1)*block] = cb

        return C
    else:
        comm.send(C_block, dest=0)
        return None