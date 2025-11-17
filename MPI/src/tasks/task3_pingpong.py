from mpi4py import MPI
import numpy as np
import time


def run_task3_pingpong(n: int, find_min: bool = True, comm: MPI.Comm = MPI.COMM_WORLD):
    """
    Классический ping-pong между двумя процессами.
    Работает только при mpirun -n 2.
    Измеряет среднее время одной передачи сообщения длиной n байт.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 2:
        if rank == 0:
            print("❌ Для этой задачи нужно ровно 2 процесса (mpirun -n 2)")
        return None

    msg = np.ones(n, dtype='b')

    comm.Barrier()
    t0 = time.perf_counter()

    for _ in range(100):  # повторов достаточно много, но не слишком
        if rank == 0:
            comm.Send([msg, MPI.BYTE], dest=1)
            comm.Recv([msg, MPI.BYTE], source=1)
        else:
            comm.Recv([msg, MPI.BYTE], source=0)
            comm.Send([msg, MPI.BYTE], dest=0)

    comm.Barrier()
    t1 = time.perf_counter()

    avg_time = (t1 - t0) / (2 * 100)  # однонаправленное среднее время
    if rank == 0:
        return avg_time
    return None