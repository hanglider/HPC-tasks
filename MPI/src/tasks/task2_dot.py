from mpi4py import MPI
import numpy as np

def run_task2_dot(vector_size: int, comm=MPI.COMM_WORLD) -> float:
    rank = comm.Get_rank()
    size = comm.Get_size()

    # равномерное распределение
    base = vector_size // size
    rem = vector_size % size
    counts = np.array([base + (1 if r < rem else 0) for r in range(size)], dtype=np.int32)
    displs = np.array([int(np.sum(counts[:r])) for r in range(size)], dtype=np.int32)

    if rank == 0:
        a = np.random.rand(vector_size)
        b = np.random.rand(vector_size)
    else:
        a = b = None

    local_a = np.empty(counts[rank], dtype=np.float64)
    local_b = np.empty(counts[rank], dtype=np.float64)

    comm.Scatterv([a, counts, displs, MPI.DOUBLE], local_a, root=0)
    comm.Scatterv([b, counts, displs, MPI.DOUBLE], local_b, root=0)

    local_sum = np.dot(local_a, local_b)

    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    return float(global_sum)