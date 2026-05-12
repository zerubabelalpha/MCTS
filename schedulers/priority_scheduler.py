from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

from environment.task import Status

if TYPE_CHECKING:
    from environment.scheduler_env import SchedulerEnvironment


class PriorityScheduler:
    """
    Priority-first, earliest-deadline-first within each priority band.
    """

    def __init__(self):
        self.name = "Priority"

    def schedule(self, env: "SchedulerEnvironment") -> List[Tuple]:
        """Return one assign action per free worker (highest urgency first)."""
        actions = []
        waiting = [t for t in env.queue if t.status == Status.WAITING]

        # Sort: highest priority first; break ties by nearest deadline
        sorted_tasks = sorted(
            waiting,
            key=lambda t: (-t.priority.value, t.deadline),
        )
        free_ids = [w.worker_id for w in env.workers if w.is_free]

        for wid, task in zip(free_ids, sorted_tasks):
            idx = env.queue.index(task)
            actions.append(('assign', idx, wid))

        return actions if actions else [('idle',)]
