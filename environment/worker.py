from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.task import Task


class Worker:
    """
    A worker processes one Task at a time.

    States
    ------
    free    : current_task is None and not failed
    busy    : current_task is not None
    failed  : temporarily offline (recovery_ticks > 0)
    """

    def __init__(self, worker_id: int):
        self.worker_id      : int          = worker_id
        self.current_task   : Task | None  = None
        self.failed         : bool         = False
        self.recovery_ticks : int          = 0   # ticks until back online

        # Running statistics
        self.total_ticks_busy : int = 0
        self.tasks_completed  : int = 0

    #  state queries 

    @property
    def is_free(self) -> bool:
        return not self.failed and self.current_task is None

    @property
    def is_busy(self) -> bool:
        return not self.failed and self.current_task is not None

    #  operations 

    def assign(self, task: "Task") -> None:
        """Assign a task to this worker (caller must check is_free first)."""
        self.current_task       = task
        task.assigned_to        = self.worker_id

    def tick(self) -> "Task | None":
       
        # --- handle failure recovery ---
        if self.failed:
            self.recovery_ticks -= 1
            if self.recovery_ticks <= 0:
                self.failed         = False
                self.recovery_ticks = 0
            return None

        # --- advance current task ---
        if self.current_task is not None:
            self.total_ticks_busy += 1
            self.current_task.remaining -= 1

            if self.current_task.remaining <= 0:
                finished          = self.current_task
                self.current_task = None
                self.tasks_completed += 1
                return finished

        return None

    def fail(self, recovery_ticks: int) -> None:
        """Put the worker offline for `recovery_ticks` ticks."""
        self.failed         = True
        self.recovery_ticks = recovery_ticks
        # drop current task back to queue (handled by scheduler_env)
        self.current_task   = None

    def __repr__(self) -> str:
        state = "FAILED" if self.failed else ("BUSY" if self.is_busy else "FREE")
        return f"Worker({self.worker_id}, {state}, task={self.current_task})"
