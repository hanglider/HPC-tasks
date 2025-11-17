from .task1_minmax import run_task1_minmax
from .task2_dot import run_task2_dot
from .task4_striped_mm import run_task4_striped_mm
from .task4_fox_mm import run_task4_fox_mm

TASKS = {
    "task1_minmax": run_task1_minmax,
    "task2_dot": run_task2_dot,
    "task4_striped_mm": run_task4_striped_mm,
    "task4_fox_mm": run_task4_fox_mm,
}