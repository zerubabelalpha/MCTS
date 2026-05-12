from __future__ import annotations

import random
from typing import List, Tuple, TYPE_CHECKING

from config import (
    MCTS_ROLLOUT_COUNT,
    MCTS_EXPLORATION_C,
    MCTS_ROLLOUT_DEPTH,
    REWARD_IDLE_WORKER,
    REWARD_QUEUE_CONGESTION,
    QUEUE_CONGESTION_THRESHOLD,
)
from environment.task import Status, Priority
from mcts.node import MCTSNode

if TYPE_CHECKING:
    from environment.scheduler_env import SchedulerEnvironment


class MCTSScheduler:
    

    def __init__(self):
        self.name = "MCTS"

    # ── public interface ──────────────────────────────────────────────────────

    def schedule(self, env: "SchedulerEnvironment") -> List[Tuple]:
        """
        Run MCTS from the current environment state.

        Returns a *list* of actions (one per free worker with a task to assign).
        MCTS plans one action at a time — we greedily chain assignments until
        no free workers or waiting tasks remain.
        """
        assigned_actions = []

        # Greedily assign tasks to all free workers via separate MCTS calls
        # (each call sees the updated snapshot with previous assignments)
        sim_env = env.snapshot()

        while sim_env.num_free_workers > 0 and sim_env.waiting_tasks:
            action = self._run_mcts(sim_env)
            if action[0] == 'idle':
                break
            assigned_actions.append(action)
            sim_env.apply_action(action)   # update snapshot for next iteration

        return assigned_actions if assigned_actions else [('idle',)]

    #  MCTS core 

    def _run_mcts(self, env: "SchedulerEnvironment") -> Tuple:
        """Run MCTS iterations and return the best single action."""
        root = MCTSNode()
        root._unexpanded = env.get_valid_actions()

        # Edge case: only one action available
        if len(root._unexpanded) == 1:
            return root._unexpanded[0]

        for _ in range(MCTS_ROLLOUT_COUNT):
            #  1. Selection 
            node, sim = self._select(root, env)

            #  2. Expansion 
            node, sim = self._expand(node, sim)

            #  3. Simulation (rollout) 
            value = self._simulate(sim)

            #  4. Backpropagation 
            self._backpropagate(node, value)

        # Return action of the most-visited child
        if not root.children:
            return ('idle',)
        return root.best_action()

    #  selection 

    def _select(
        self,
        node: MCTSNode,
        env : "SchedulerEnvironment",
    ) -> Tuple[MCTSNode, "SchedulerEnvironment"]:
        """
        Walk down the tree following UCB1 until we find a non-fully-expanded
        or leaf node. Returns the node and a snapshot of the env at that node.
        """
        sim = env.snapshot()

        while not node.is_leaf() and node.is_fully_expanded():
            node = node.best_child(MCTS_EXPLORATION_C)
            if node.parent_action and node.parent_action[0] != 'idle':
                sim.apply_action(node.parent_action)

        return node, sim

    #  expansion 

    def _expand(
        self,
        node: MCTSNode,
        sim : "SchedulerEnvironment",
    ) -> Tuple[MCTSNode, "SchedulerEnvironment"]:
        """
        Add one unexplored child.  If this is the first visit, initialise
        _unexpanded from the current sim's valid actions.
        """
        if node._unexpanded is None:
            node._unexpanded = sim.get_valid_actions()

        if not node._unexpanded:
            return node, sim

        # Choose an unexplored action (heuristic: prefer urgent tasks)
        action = self._heuristic_pick(node._unexpanded, sim)
        child  = node.add_child(action)

        # Apply the action to the rollout snapshot
        new_sim = sim.snapshot()
        if action[0] != 'idle':
            new_sim.apply_action(action)

        return child, new_sim

    #  simulation (rollout) 

    def _simulate(self, sim: "SchedulerEnvironment") -> float:
        """
        Heuristic rollout: simulate MCTS_ROLLOUT_DEPTH steps, assigning tasks
        greedily by urgency score at each step.
        Returns the cumulative reward collected.
        """
        total_reward = 0.0
        rollout_env  = sim.snapshot()

        for _ in range(MCTS_ROLLOUT_DEPTH):
            # Heuristic action: assign most-urgent task to each free worker
            actions = self._heuristic_actions(rollout_env)
            reward  = rollout_env.step(actions)
            total_reward += reward

        return total_reward

    #  backpropagation 

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Propagate value up the tree to the root."""
        while node is not None:
            node.update(value)
            node = node.parent

    #  heuristic helpers 

    def _heuristic_pick(
        self,
        actions  : List[Tuple],
        env      : "SchedulerEnvironment",
    ) -> Tuple:
        """
        Select one action from the unexplored list.
        Prefer assigning higher-urgency tasks; fall back to 'idle'.
        """
        assign_actions = [a for a in actions if a[0] == 'assign']
        if not assign_actions:
            return ('idle',)

        # Score each assign action by the urgency of the task it assigns
        def score(action):
            _, task_idx, _ = action
            if task_idx >= len(env.queue):
                return 0.0
            task = env.queue[task_idx]
            return task.urgency_score(env.tick)

        return max(assign_actions, key=score)

    def _heuristic_actions(
        self,
        env: "SchedulerEnvironment",
    ) -> List[Tuple]:
        """
        Build a greedy action list for rollout steps:
        assign the most-urgent waiting task to each free worker.
        """
        actions   = []
        waiting   = sorted(
            env.waiting_tasks,
            key=lambda t: t.urgency_score(env.tick),
            reverse=True,
        )
        free_wids = [w.worker_id for w in env.workers if w.is_free]

        for wid, task in zip(free_wids, waiting):
            idx = env.queue.index(task)
            actions.append(('assign', idx, wid))

        return actions if actions else [('idle',)]
