# Adaptive Dynamic Task Scheduling Using Monte Carlo Tree Search

>  [problem statment](https://docs.google.com/document/d/1sz2WPpC8t0Kf-uFQaVdgr8xVSBi7BjjLfBIbKUQPqTI/edit?usp=sharing)
>  [The Four loops](https://docs.google.com/document/d/1TGMiA3HHQqc0LSGlHI_K7glODNmTqjShlPSRQQGx4oE/edit?usp=sharing)
>  [why mcts ?](https://docs.google.com/document/d/1U_F2LXZ76slFiQZVdKtY9c4jshr03OWxRP98lK2l_tI/edit?usp=sharing)


A lightweight, modular scheduling simulator where an MCTS-based scheduler dynamically assigns tasks to workers and is compared head-to-head against FIFO and Priority baselines — all visualised in real-time via Pygame.


## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the interactive visualiser (MCTS by default)

```bash
python main.py
```

### 3. Start with a specific scheduler

```bash
python main.py --scheduler FIFO
python main.py --scheduler Priority
python main.py --scheduler MCTS
```

### 4. Run the headless benchmark (all 3 schedulers compared)

```bash
python main.py --bench
```

## How It Works

### Simulation Loop (per tick)

```
1. Generate dynamic events   → new task arrivals, priority escalation, worker failures
2. Scheduler decides actions → which task goes to which free worker
3. Apply actions             → bind tasks to workers
4. Advance workers           → decrement remaining ticks, collect completions
5. Check deadlines           → mark overdue tasks as FAILED
6. Update metrics            → KPIs accumulated in MetricsTracker
7. Render                    → Pygame draws queue, workers, metrics bar
```

### MCTS Algorithm

Each tick the MCTS scheduler runs `MCTS_ROLLOUT_COUNT` (default 80) iterations:

```
For each rollout:
  SELECT     Walk tree via UCB1 until a non-fully-expanded node is found
  EXPAND     Add one unexplored child (heuristic: prefer urgent tasks)
  SIMULATE   Run MCTS_ROLLOUT_DEPTH heuristic steps, accumulate reward
  BACKPROP   Propagate total reward up to root
Return the action of the most-visited root child.
```

The **rollout policy** is not random — it greedily assigns the most-urgent task (highest urgency score = priority × 10 / deadline_slack) to each free worker.

### Reward Function

| Event                              | Reward          |
|------------------------------------|-----------------|
| Task completed                     | `+1.0`          |
| HIGH / CRITICAL task completed     | `+1.5` bonus    |
| Deadline missed                    | `−3.0`          |
| Per idle worker per tick           | `−0.1`          |
| Per task above queue threshold (6) | `−0.05`         |

### Dynamic Events

| Event                    | Probability (per tick) | Config key                  |
|--------------------------|------------------------|-----------------------------|
| New task arrival         | 35%                    | `TASK_ARRIVAL_PROB`         |
| Priority escalation      | 4% per waiting task    | `PRIORITY_ESCALATION_PROB`  |
| Worker failure           | 0.3% per worker        | `WORKER_FAILURE_PROB`       |


## Tracked Metrics

1. **Total completed tasks**
2. **Average waiting time** (ticks from arrival to execution start)
3. **Deadline miss rate** (failed / total processed)
4. **Worker utilisation** (fraction of ticks each worker is busy)
5. **Average queue length** (waiting tasks per tick)
6. **Cumulative reward** (sum of all step rewards)

