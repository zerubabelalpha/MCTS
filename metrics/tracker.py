from __future__ import annotations

from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from environment.scheduler_env import SchedulerEnvironment


class MetricsTracker:


    def __init__(self, num_workers: int):
        self._num_workers = num_workers

        # Per-tick snapshots
        self._queue_lengths    : List[int]   = []
        self._idle_counts      : List[int]   = []
        self._rewards          : List[float] = []

        # Accumulated totals
        self.total_completed   : int   = 0
        self.total_failed      : int   = 0
        self.total_waiting_time: float = 0.0   # sum of (start_tick - arrival)
        self.total_ticks       : int   = 0

        # Snapshot helpers (previous tick's counts to detect new events)
        self._prev_completed   : int = 0
        self._prev_failed      : int = 0

    #  per-tick recording 

    def record(self, env: "SchedulerEnvironment", reward: float) -> None:
        
        self.total_ticks += 1

        # Queue length (waiting only)
        self._queue_lengths.append(
            len(env.waiting_tasks)
        )

        # Idle worker count
        idle = sum(1 for w in env.workers if w.is_free)
        self._idle_counts.append(idle)

        # Reward
        self._rewards.append(reward)

        # New completions this tick
        new_done = len(env.completed) - self._prev_completed
        self.total_completed += new_done
        self._prev_completed  = len(env.completed)

        # New failures this tick
        new_fail = len(env.failed) - self._prev_failed
        self.total_failed += new_fail
        self._prev_failed    = len(env.failed)

        # Waiting time: for each task that just completed, record how long
        # it waited in queue (arrival → actual execution start, approximated
        # as current_tick - arrival_time - duration)
        for task in env.completed[self._prev_completed - new_done : self._prev_completed]:
            wait = max(0, env.tick - task.arrival_time - task.duration)
            self.total_waiting_time += wait

   

    def summary(self, scheduler_name: str = "Unknown") -> Dict[str, Any]:
        """Return a dictionary of aggregated metrics."""
        total_tasks = self.total_completed + self.total_failed

        miss_rate = (
            self.total_failed / total_tasks
            if total_tasks > 0 else 0.0
        )
        avg_wait = (
            self.total_waiting_time / self.total_completed
            if self.total_completed > 0 else 0.0
        )
        avg_queue = (
            sum(self._queue_lengths) / len(self._queue_lengths)
            if self._queue_lengths else 0.0
        )
        avg_idle = (
            sum(self._idle_counts) / len(self._idle_counts)
            if self._idle_counts else 0.0
        )
        utilisation = (
            1.0 - avg_idle / self._num_workers
            if self._num_workers > 0 else 0.0
        )
        cum_reward = sum(self._rewards)

        return {
            "scheduler"         : scheduler_name,
            "total_completed"   : self.total_completed,
            "total_failed"      : self.total_failed,
            "deadline_miss_rate": round(miss_rate,    4),
            "avg_waiting_time"  : round(avg_wait,     2),
            "worker_utilisation": round(utilisation,  4),
            "avg_queue_length"  : round(avg_queue,    2),
            "cumulative_reward" : round(cum_reward,   2),
            "total_ticks"       : self.total_ticks,
        }

    def print_summary(self, scheduler_name: str = "Unknown") -> None:
        """Pretty-print the summary table."""
        s = self.summary(scheduler_name)
        print(f"\n{'═'*52}")
        print(f"  Scheduler : {s['scheduler']}")
        print(f"{'─'*52}")
        print(f"  Completed tasks    : {s['total_completed']}")
        print(f"  Failed (missed dl) : {s['total_failed']}")
        print(f"  Deadline miss rate : {s['deadline_miss_rate']*100:.1f}%")
        print(f"  Avg waiting time   : {s['avg_waiting_time']:.1f} ticks")
        print(f"  Worker utilisation : {s['worker_utilisation']*100:.1f}%")
        print(f"  Avg queue length   : {s['avg_queue_length']:.1f}")
        print(f"  Cumulative reward  : {s['cumulative_reward']:.1f}")
        print(f"{'═'*52}\n")
