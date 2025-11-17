from mpi4py import MPI
import numpy as np
import math

def run_task4_fox_mm(n: int, comm: MPI.Comm = MPI.COMM_WORLD):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Fox требует квадратную решётку процессов
    p = int(math.sqrt(size))
    if p * p != size:
        if rank == 0:
            print(f"❌ Fox algorithm requires p^2 processes (4,9,16,...). Запущено {size}")
        return None

    # Создаём 2D-карты
    dims = [p, p]
    periods = [0, 0]
    grid = comm.Create_cart(dims, periods=periods, reorder=True)
    coords = grid.Get_coords(rank)
    row_comm = grid.Sub([1, 0])
    col_comm = grid.Sub([0, 1])

    block = n // p  # квадратная блок-матрица

    # Root генерирует A,B
    if rank == 0:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
    else:
        A = None
        B = None

    # Разбиваем матрицы на блоки
    A_block = np.zeros((block, block))
    B_block = np.zeros((block, block))

    if rank == 0:
        # режем на p×p блоков
        for i in range(p):
            for j in range(p):
                dest = i * p + j
                Ab = A[i*block:(i+1)*block, j*block:(j+1)*block]
                Bb = B[i*block:(i+1)*block, j*block:(j+1)*block]
                if dest == 0:
                    A_block[:, :] = Ab
                    B_block[:, :] = Bb
                else:
                    comm.send((Ab, Bb), dest=dest)
    else:
        Ab, Bb = comm.recv(source=0)
        A_block[:, :] = Ab
        B_block[:, :] = Bb

    C_block = np.zeros((block, block))

    # Алгоритм Фокса
    for k in range(p):
        # Определяем блок A ("выбираем ведущий блок")
        if coords[1] == k:
            broadcast_block = A_block.copy()
        else:
            broadcast_block = np.empty((block, block))

        # рассылаем блок по строке
        row_comm.Bcast(broadcast_block, root=k)

        # локальное умножение
        C_block += broadcast_block @ B_block

        # циклический сдвиг B вверх
        B_next = np.empty_like(B_block)
        col_comm.Sendrecv(B_block, B_next, dest=(coords[0] - 1) % p, source=(coords[0] + 1) % p)
        B_block = B_next

    # Собираем результат
    if rank == 0:
        C = np.zeros((n, n))
        # Вставляем свой блок
        C[0:block, 0:block] = C_block

        for r in range(1, size):
            cb = comm.recv(source=r)
            i, j = divmod(r, p)
            C[i*block:(i+1)*block, j*block:(j+1)*block] = cb

        return C
    else:
        comm.send(C_block, dest=0)
        return None