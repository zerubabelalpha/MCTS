from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

from environment.task import Status

if TYPE_CHECKING:
    from environment.scheduler_env import SchedulerEnvironment


class FIFOScheduler:
    """
    Assigns the oldest waiting task to each free worker.
    """

    def __init__(self):
        self.name = "FIFO"

    def schedule(self, env: "SchedulerEnvironment") -> List[Tuple]:
        """Return one assign action per free worker (if tasks available)."""
        actions  = []
        # Tasks in queue are already ordered by insertion (arrival) time
        waiting  = [t for t in env.queue if t.status == Status.WAITING]
        free_ids = [w.worker_id for w in env.workers if w.is_free]

        for wid, task in zip(free_ids, waiting):
            idx = env.queue.index(task)
            actions.append(('assign', idx, wid))

        return actions if actions else [('idle',)]
