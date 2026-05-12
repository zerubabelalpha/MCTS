
#  Simulation 
SIMULATION_DURATION     = 300        # total timesteps before auto-stop
TIMESTEP_DELAY_MS       = 120        # milliseconds between ticks (Pygame clock)

#  Workers 
NUM_WORKERS             = 4          # how many workers run in parallel
WORKER_FAILURE_PROB     = 0.003      # probability a worker fails each tick
WORKER_RECOVERY_TICKS   = 15        # ticks a failed worker stays offline

#  Task generation 
TASK_ARRIVAL_PROB       = 0.35       # P(new task arrives each tick)
MAX_QUEUE_SIZE          = 20         # hard cap on waiting queue length
INITIAL_TASKS           = 6         # tasks pre-loaded at t=0

# Task duration range (ticks)
TASK_MIN_DURATION       = 5
TASK_MAX_DURATION       = 25

# Task deadline range (ticks from arrival)
TASK_MIN_DEADLINE       = 10
TASK_MAX_DEADLINE       = 60

#  Dynamic events 
PRIORITY_ESCALATION_PROB = 0.04     # P(a waiting task gets escalated each tick)

#  MCTS 
MCTS_ROLLOUT_COUNT      = 80        # simulated rollouts per decision
MCTS_EXPLORATION_C      = 1.41      # UCB1 exploration constant (√2 ≈ 1.41)
MCTS_ROLLOUT_DEPTH      = 8         # how many steps each rollout simulates

#  Reward weights 
REWARD_TASK_COMPLETE        =  1.0
REWARD_HIGH_PRIORITY_BONUS  =  1.5   # extra reward for HIGH / CRITICAL completion
REWARD_DEADLINE_MISS        = -3.0
REWARD_IDLE_WORKER          = -0.1   # per idle worker per tick
REWARD_QUEUE_CONGESTION     = -0.05  # per task in queue above threshold
QUEUE_CONGESTION_THRESHOLD  =  6

#  Pygame window 
WINDOW_WIDTH            = 1200
WINDOW_HEIGHT           = 700
FPS                     = 30
FONT_SIZE_LARGE         = 20
FONT_SIZE_MEDIUM        = 16
FONT_SIZE_SMALL         = 13
