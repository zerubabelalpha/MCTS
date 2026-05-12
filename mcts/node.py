from __future__ import annotations

import math
from typing import List, Optional, Tuple


class MCTSNode:
   

    def __init__(
        self,
        parent       : Optional["MCTSNode"] = None,
        parent_action: Optional[Tuple]      = None,
    ):
        self.parent        = parent
        self.parent_action = parent_action

        self.children      : List["MCTSNode"] = []
        self.visit_count   : int   = 0
        self.total_value   : float = 0.0

        # Actions not yet expanded from this node
        self._unexpanded   : Optional[List[Tuple]] = None

    #  UCB1 score 

    def ucb1(self, exploration_c: float) -> float:
        
        if self.visit_count == 0:
            return float('inf')

        exploitation = self.total_value / self.visit_count
        exploration  = exploration_c * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        return exploitation + exploration

    #  tree helpers 

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_fully_expanded(self) -> bool:
        return self._unexpanded is not None and len(self._unexpanded) == 0

    def best_child(self, exploration_c: float) -> "MCTSNode":
        """Return the child with the highest UCB1 score."""
        return max(self.children, key=lambda c: c.ucb1(exploration_c))

    def best_action(self) -> Tuple:
        """Return the action of the most-visited child (exploitation only)."""
        return max(self.children, key=lambda c: c.visit_count).parent_action

    def add_child(self, action: Tuple) -> "MCTSNode":
        """Create and register a new child for the given action."""
        child = MCTSNode(parent=self, parent_action=action)
        self.children.append(child)
        if self._unexpanded and action in self._unexpanded:
            self._unexpanded.remove(action)
        return child

    def update(self, value: float) -> None:
        """Backpropagate: increment visit count and accumulate reward."""
        self.visit_count += 1
        self.total_value += value

    def __repr__(self) -> str:
        q = self.total_value / self.visit_count if self.visit_count else 0
        return (
            f"MCTSNode(action={self.parent_action}, "
            f"visits={self.visit_count}, Q={q:.3f})"
        )
