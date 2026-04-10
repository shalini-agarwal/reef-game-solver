from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass

from game_simulation.actions import BaseAction, ClaimWinAction
from game_simulation.game_state import GameState
from game_simulation.game_types import ResourceType
from game_simulation.structure_mixins import Structure


@dataclass
class ExpansionOption:
    """Represents a potential expansion to a resource node."""

    x: int
    y: int
    resource_type: ResourceType
    structure: Structure


class StrategyFailed(Exception):
    """Exception raised when a strategy branch fails."""

    pass


class BaseStrategy(ABC):
    """Abstract base class for all game strategies."""

    ACTION_LOOP_LIMIT = 100

    def __init__(self, game_state: GameState):
        self.game_state = game_state
        self.plan: list[list[BaseAction]] = []

        # Configure logger with [Strategy] prefix
        self.strategy = logging.getLogger("strategy")
        if not self.strategy.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[Strategy] %(message)s")
            handler.setFormatter(formatter)
            self.strategy.addHandler(handler)
            self.strategy.setLevel(logging.DEBUG)

    def generate_plan(self) -> None:
        """Generate the complete plan to win the game."""
        while self.game_state.max_turns > len(self.plan):
            self.game_state.executed_turns = len(self.plan)
            self.generate_next_turn()

            if not self.plan or not self.plan[-1]:
                if self.plan:
                    self.plan.pop()
                raise StrategyFailed(f"No actions generated for turn {self.game_state.executed_turns + 1}.")

            last_turn = self.plan[-1]
            if any(isinstance(action, ClaimWinAction) for action in last_turn):
                return

        raise StrategyFailed(f"Maximum number of turns reached ({self.game_state.max_turns}).")

    def generate_next_turn(self) -> None:
        self.game_state.turn_start()
        new_turn: list[BaseAction] = []
        self.plan.append(new_turn)

        for _ in range(self.ACTION_LOOP_LIMIT):
            loop_actions: list[BaseAction] = []

            for action in self.generate_more_turn_actions():
                loop_actions.append(action)
                action.apply(self.game_state)

            if not loop_actions:
                break
            new_turn.extend(loop_actions)
        else:
            raise StrategyFailed(
                "Reached BaseStrategy.ACTION_LOOP_LIMIT while generating a turn. "
                "Infinite loop prevention triggered; increase the constant if more actions are required."
            )

    @abstractmethod
    def generate_more_turn_actions(self) -> Generator[BaseAction]: ...
