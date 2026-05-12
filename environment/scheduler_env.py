from __future__ import annotations

import copy
import random
from typing import List, Optional, Tuple

import numpy as np

from config import (
    NUM_WORKERS,
    TASK_ARRIVAL_PROB,
    MAX_QUEUE_SIZE,
    INITIAL_TASKS,
    TASK_MIN_DURATION,
    TASK_MAX_DURATION,
    TASK_MIN_DEADLINE,
    TASK_MAX_DEADLINE,
    PRIORITY_ESCALATION_PROB,
    WORKER_FAILURE_PROB,
    WORKER_RECOVERY_TICKS,
    REWARD_TASK_COMPLETE,
    REWARD_HIGH_PRIORITY_BONUS,
    REWARD_DEADLINE_MISS,
    REWARD_IDLE_WORKER,
    REWARD_QUEUE_CONGESTION,
    QUEUE_CONGESTION_THRESHOLD,
)
from environment.task import Task, Priority, Status
from environment.worker import Worker


# Priority weights used when sampling random task priorities
_PRIORITY_WEIGHTS = [0.30, 0.35, 0.25, 0.10]  # LOW, MEDIUM, HIGH, CRITICAL


def _random_priority() -> Priority:
    return random.choices(list(Priority), weights=_PRIORITY_WEIGHTS, k=1)[0]


def _make_task(current_tick: int) -> Task:
    """Factory for a randomly-generated task arriving at current_tick."""
    duration = random.randint(TASK_MIN_DURATION, TASK_MAX_DURATION)
    deadline = current_tick + random.randint(TASK_MIN_DEADLINE, TASK_MAX_DEADLINE)
    return Task(
        duration     = duration,
        priority     = _random_priority(),
        deadline     = deadline,
        arrival_time = current_tick,
    )


class SchedulerEnvironment:
    """
    Core simulation environment.

    Public interface

    reset()                       → initialise / restart
    step(actions)                 → advance one tick, return reward
    get_valid_actions()           → list of (task_idx, worker_idx) or ('idle',)
    snapshot() / restore(snap)    → deep-copy helpers for MCTS rollouts
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self.reset()

    #  initialisation 

    def reset(self) -> None:
        random.seed(self._seed)
        np.random.seed(self._seed)

        self.tick       : int  = 0
        self.done       : bool = False
        self.cum_reward : float = 0.0

        # Workers
        self.workers: List[Worker] = [Worker(i) for i in range(NUM_WORKERS)]

        # Queues
        self.queue     : List[Task] = []   # waiting tasks
        self.completed : List[Task] = []   # successfully finished
        self.failed    : List[Task] = []   # missed deadline

        # Pre-load initial tasks
        for _ in range(INITIAL_TASKS):
            self.queue.append(_make_task(self.tick))

    #  public interface 

    def get_valid_actions(self) -> List[Tuple]:
        """
        Return a list of possible actions for this tick.

        Each action is one of:
            ('assign', task_idx, worker_idx)   — assign queue[task_idx] to workers[worker_idx]
            ('idle',)                           — do nothing

        Only free workers and waiting tasks are considered.
        """
        free_workers  = [w for w in self.workers if w.is_free]
        waiting_tasks = [t for t in self.queue if t.status == Status.WAITING]

        actions = [('idle',)]
        for w in free_workers:
            for i, t in enumerate(waiting_tasks):
                actions.append(('assign', self.queue.index(t), w.worker_id))

        return actions

    def apply_action(self, action: Tuple) -> None:
        """Apply a single action without advancing the clock."""
        if action[0] == 'idle':
            return
        _, task_idx, worker_id = action
        if task_idx >= len(self.queue):
            return
        task   = self.queue[task_idx]
        worker = self.workers[worker_id]
        if task.status == Status.WAITING and worker.is_free:
            task.status = Status.RUNNING
            worker.assign(task)

    def step(self, actions: List[Tuple]) -> float:
        """
        Advance the simulation by one tick.

        1. Generate dynamic events (arrivals, escalations, failures).
        2. Apply the scheduler's chosen actions.
        3. Advance workers (decrement remaining, collect completions).
        4. Check deadlines.
        5. Compute and return the step reward.
        """
        reward = 0.0

        #  1. Dynamic events 
        reward += self._generate_events()

        #  2. Apply actions 
        for action in actions:
            self.apply_action(action)

        #  3. Advance workers 
        for worker in self.workers:
            finished = worker.tick()
            if finished is not None:
                # Guard: the task may have already been expired and removed from
                # the queue by a previous tick's deadline check.  In that case
                # it is already marked FAILED — skip it silently.
                if finished.status == Status.FAILED:
                    continue

                finished.status = Status.COMPLETED
                if finished in self.queue:          # defensive – should always be True here
                    self.queue.remove(finished)
                self.completed.append(finished)

                # Base completion reward
                reward += REWARD_TASK_COMPLETE
                # Bonus for high-priority completions
                if finished.priority in (Priority.HIGH, Priority.CRITICAL):
                    reward += REWARD_HIGH_PRIORITY_BONUS

        #  4. Deadline checks 
        self.tick += 1
        expired = [
            t for t in self.queue
            if t.status in (Status.WAITING, Status.RUNNING)
               and t.is_overdue(self.tick)
        ]
        for t in expired:
            t.status = Status.FAILED
            self.queue.remove(t)
            self.failed.append(t)
            reward += REWARD_DEADLINE_MISS
            # If the task was RUNNING, free the worker that held it so
            # that worker.tick() won't return a stale reference next tick.
            for w in self.workers:
                if w.current_task is t:
                    w.current_task = None

        #  5. Idle-worker and queue-congestion penalties 
        idle_count = sum(1 for w in self.workers if w.is_free)
        reward += idle_count * REWARD_IDLE_WORKER

        queue_len  = len([t for t in self.queue if t.status == Status.WAITING])
        if queue_len > QUEUE_CONGESTION_THRESHOLD:
            reward += (queue_len - QUEUE_CONGESTION_THRESHOLD) * REWARD_QUEUE_CONGESTION

        self.cum_reward += reward
        return reward

    #  dynamic event helpers 

    def _generate_events(self) -> float:
        """Spawn arrivals, escalate priorities, and optionally fail workers."""
        reward = 0.0

        # New task arrival
        if (len(self.queue) < MAX_QUEUE_SIZE
                and random.random() < TASK_ARRIVAL_PROB):
            self.queue.append(_make_task(self.tick))

        # Priority escalation
        waiting = [t for t in self.queue if t.status == Status.WAITING]
        for task in waiting:
            if (task.priority != Priority.CRITICAL
                    and random.random() < PRIORITY_ESCALATION_PROB):
                # bump one level
                new_level = task.priority.value + 1
                task.priority = Priority(new_level)

        # Worker failure (optional dynamic event)
        for worker in self.workers:
            if (not worker.failed
                    and random.random() < WORKER_FAILURE_PROB):
                # Return its task to queue if any
                if worker.current_task is not None:
                    worker.current_task.status      = Status.WAITING
                    worker.current_task.assigned_to = None
                worker.fail(WORKER_RECOVERY_TICKS)

        return reward

    #  MCTS snapshot helpers 

    def snapshot(self) -> "SchedulerEnvironment":
        """
        Return a deep copy of this environment for MCTS rollouts.
        The copy uses a different random seed to avoid replay artefacts.
        """
        snap = copy.deepcopy(self)
        snap._seed = random.randint(0, 10_000)
        return snap

    def restore(self, snap: "SchedulerEnvironment") -> None:
        """Restore live state from a snapshot (used after MCTS planning)."""
        self.__dict__.update(copy.deepcopy(snap.__dict__))

    #  informational helpers 

    @property
    def waiting_tasks(self) -> List[Task]:
        return [t for t in self.queue if t.status == Status.WAITING]

    @property
    def running_tasks(self) -> List[Task]:
        return [t for t in self.queue if t.status == Status.RUNNING]

    @property
    def num_free_workers(self) -> int:
        return sum(1 for w in self.workers if w.is_free)

    def __repr__(self) -> str:
        return (
            f"Env(tick={self.tick}, queue={len(self.queue)}, "
            f"completed={len(self.completed)}, failed={len(self.failed)})"
        )
