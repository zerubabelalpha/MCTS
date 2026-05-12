import sys
import time
import argparse

import pygame

from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, FPS,
    SIMULATION_DURATION, TIMESTEP_DELAY_MS, NUM_WORKERS,
)
from environment.scheduler_env import SchedulerEnvironment
from schedulers.fifo_scheduler     import FIFOScheduler
from schedulers.priority_scheduler import PriorityScheduler
from mcts.mcts_scheduler           import MCTSScheduler
from metrics.tracker               import MetricsTracker
from visualization.renderer        import Renderer


#  scheduler registry 
SCHEDULERS = {
    "FIFO"    : FIFOScheduler,
    "Priority": PriorityScheduler,
    "MCTS"    : MCTSScheduler,
}


# Interactive mode

def run_interactive(scheduler_name: str = "MCTS") -> None:
    """Run the Pygame visualiser with real-time scheduler switching."""

    pygame.init()
    screen  = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Adaptive Task Scheduling — MCTS")
    clock   = pygame.time.Clock()

    def make_state(sched_name):
        env       = SchedulerEnvironment(seed=42)
        scheduler = SCHEDULERS[sched_name]()
        tracker   = MetricsTracker(num_workers=NUM_WORKERS)
        return env, scheduler, tracker

    sched_name = scheduler_name
    env, scheduler, tracker = make_state(sched_name)
    renderer = Renderer(screen)

    paused   = False
    finished_printed = False
    running  = True

    while running:
        # ... Event handling ...
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_r:
                    env, scheduler, tracker = make_state(sched_name)
                    paused = False
                    finished_printed = False

                elif event.key == pygame.K_1:
                    sched_name = "FIFO"
                    env, scheduler, tracker = make_state(sched_name)
                    finished_printed = False

                elif event.key == pygame.K_2:
                    sched_name = "Priority"
                    env, scheduler, tracker = make_state(sched_name)
                    finished_printed = False

                elif event.key == pygame.K_3:
                    sched_name = "MCTS"
                    env, scheduler, tracker = make_state(sched_name)
                    finished_printed = False

        #  Simulation step 
        if not paused and env.tick < SIMULATION_DURATION:
            actions = scheduler.schedule(env)
            reward  = env.step(actions)
            tracker.record(env, reward)

        elif env.tick >= SIMULATION_DURATION and not finished_printed:
            # Auto-pause and print summary exactly once when done
            paused = True
            finished_printed = True
            tracker.print_summary(sched_name)

        #  Render 
        renderer.render(env, tracker, sched_name, paused)
        clock.tick(FPS)

        # Honour the configurable tick delay
        if not paused:
            pygame.time.delay(TIMESTEP_DELAY_MS)

    pygame.quit()



# Headless benchmark mode

def run_benchmark() -> None:
    """
    Run all three schedulers headlessly for SIMULATION_DURATION ticks each
    (same random seed) and print a comparative summary table.
    """
    print("\n" + "═" * 60)
    print("  BENCHMARK: Adaptive Task Scheduling Comparison")
    print("═" * 60)

    results = {}

    for name, SchedulerClass in SCHEDULERS.items():
        print(f"\n  Running {name} scheduler …", end=" ", flush=True)
        t0 = time.perf_counter()

        env       = SchedulerEnvironment(seed=42)
        scheduler = SchedulerClass()
        tracker   = MetricsTracker(num_workers=NUM_WORKERS)

        for _ in range(SIMULATION_DURATION):
            actions = scheduler.schedule(env)
            reward  = env.step(actions)
            tracker.record(env, reward)

        elapsed = time.perf_counter() - t0
        print(f"done in {elapsed:.2f}s")

        summary = tracker.summary(name)
        results[name] = summary
        tracker.print_summary(name)

    #  Comparative table 
    print("\n" + "═" * 72)
    print(f"  {'Metric':<26} {'FIFO':>12} {'Priority':>12} {'MCTS':>12}")
    print("─" * 72)

    metrics_to_show = [
        ("total_completed",    "Completed tasks"),
        ("total_failed",       "Failed tasks"),
        ("deadline_miss_rate", "Miss rate"),
        ("avg_waiting_time",   "Avg wait (ticks)"),
        ("worker_utilisation", "Worker utilisation"),
        ("avg_queue_length",   "Avg queue length"),
        ("cumulative_reward",  "Cumulative reward"),
    ]

    for key, label in metrics_to_show:
        row = f"  {label:<26}"
        for name in ["FIFO", "Priority", "MCTS"]:
            v = results[name][key]
            if isinstance(v, float):
                row += f" {v:>12.3f}"
            else:
                row += f" {v:>12}"
        print(row)

    print("═" * 72 + "\n")

    #  Winner summary 
    best_reward = max(results, key=lambda n: results[n]["cumulative_reward"])
    best_miss   = min(results, key=lambda n: results[n]["deadline_miss_rate"])
    best_util   = max(results, key=lambda n: results[n]["worker_utilisation"])

    print(f"  🏆 Best cumulative reward : {best_reward}")
    print(f"  🏆 Lowest miss rate       : {best_miss}")
    print(f"  🏆 Highest utilisation    : {best_util}")
    print()


# Entry point


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive Task Scheduling using MCTS"
    )
    parser.add_argument(
        "--bench", action="store_true",
        help="Run headless benchmark (all schedulers, no GUI)",
    )
    parser.add_argument(
        "--scheduler", choices=["FIFO", "Priority", "MCTS"], default="MCTS",
        help="Initial scheduler for interactive mode (default: MCTS)",
    )
    args = parser.parse_args()

    if args.bench:
        run_benchmark()
    else:
        run_interactive(args.scheduler)


if __name__ == "__main__":
    main()
