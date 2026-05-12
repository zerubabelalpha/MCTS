from enum import Enum, auto
import itertools


class Priority(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


class Status(Enum):
    WAITING   = auto()
    RUNNING   = auto()
    COMPLETED = auto()
    FAILED    = auto()


# Thread-safe unique ID generator
_id_counter = itertools.count(1)


class Task:
  

    def __init__(
        self,
        duration: int,
        priority: Priority,
        deadline: int,
        arrival_time: int,
    ):
        self.task_id     = next(_id_counter)
        self.duration    = duration
        self.remaining   = duration
        self.priority    = priority
        self.deadline    = deadline      # absolute tick
        self.arrival_time = arrival_time
        self.status      = Status.WAITING
        self.assigned_to: int | None = None

    #  derived helpers 

    def ticks_until_deadline(self, current_tick: int) -> int:
        """Positive = slack, 0 = due now, negative = overdue."""
        return self.deadline - current_tick

    def is_overdue(self, current_tick: int) -> bool:
        return self.ticks_until_deadline(current_tick) < 0

    def urgency_score(self, current_tick: int) -> float:
        """
        Higher score → more urgent.
        Combines priority level with deadline proximity.
        """
        slack   = max(1, self.ticks_until_deadline(current_tick))
        p_value = self.priority.value        # 1–4
        return p_value * 10.0 / slack

    def __repr__(self) -> str:
        return (
            f"Task(id={self.task_id}, pri={self.priority.name}, "
            f"rem={self.remaining}/{self.duration}, "
            f"deadline_slack={self.deadline - 0}, status={self.status.name})"
        )
