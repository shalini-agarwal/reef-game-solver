"""
Game simulation for applying actions to board state.

Provides functions to apply game actions (BUILD, MINE, TRANSFER, CLAIM_WIN) to
mutable level state for visualization and simulation purposes.
"""

from __future__ import annotations

import copy
from typing import TypeVar

from api_client.models import LevelDefinition

from .game_types import ResourceType, StructureType
from .inventory import Inventory
from .structure_mixins import ExtractionStructure, StorageStructure
from .structures import (
    BaseStructure,
    GameStructure,
    StoneQuarryStructure,
    create_structure_from_api,
)

TStructure = TypeVar("TStructure", bound=GameStructure)


class GameState:
    """Mutable game state wrapper for simulation."""

    def __init__(self, level: LevelDefinition):
        self.executed_turns = 0
        self.max_turns = level.max_turns
        self.goal = level.level_goal
        self.board = level.board
        self.structures: dict[tuple[int, int], GameStructure] = {}

        # Convert API structures to concrete structures
        for struct in level.structures:
            try:
                structure_type = StructureType(struct.type)
                concrete_struct = create_structure_from_api(struct, structure_type)
                self.structures[(struct.x, struct.y)] = concrete_struct
            except ValueError:
                # Skip unknown structure types
                continue

    def get_structures(self, structure_type: type[TStructure]) -> list[TStructure]:
        return [s for s in self.structures.values() if isinstance(s, structure_type)]

    @property
    def turns_left(self) -> int:
        return self.max_turns - self.executed_turns

    @property
    def base(self) -> BaseStructure:
        """Find the base structure in the game state."""
        for structure in self.structures.values():
            if isinstance(structure, BaseStructure):
                return structure
        raise RuntimeError("No base structure found in game state")

    @property
    def quarries(self) -> list[StoneQuarryStructure]:
        return self.get_structures(StoneQuarryStructure)

    def non_base_storage_resources(self) -> Inventory:
        """Aggregate resources currently held in storage structures other than the base."""
        non_base_storages = [
            structure.storage
            for structure in self.structures.values()
            if isinstance(structure, StorageStructure) and not isinstance(structure, BaseStructure)
        ]

        return Inventory.total(non_base_storages)

    def add_structure(self, structure: GameStructure) -> None:
        """Add a concrete structure to the game state."""
        self.structures[(structure.x, structure.y)] = structure

    def get_structure_at(self, x: int, y: int) -> GameStructure | None:
        """Get concrete structure at the specified coordinates."""
        return self.structures.get((x, y))

    def copy(self) -> GameState:
        """Create a deep copy of this game state."""
        # Deep copy the level definition
        return copy.deepcopy(self)

    @property
    def goal_resources(self) -> Inventory:
        """Get target resources as an Inventory object."""
        resources = {}
        for resource_str, amount in self.goal.target_resources.items():
            try:
                resource_type = ResourceType(resource_str)
                resources[resource_type] = amount
            except ValueError:
                # Skip unknown resource types
                continue
        return Inventory(resources)

    def turn_start(self) -> None:
        """
        Prepare the game state for the turn.
        """
        for extractable in self.get_structures(ExtractionStructure):
            extractable.extracted_this_turn = False
