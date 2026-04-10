"""
Game actions for the Recruitment Game Server.

Provides strongly typed action classes for game simulation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api_client.models import Action

    from game_simulation.game_state import GameState

from game_simulation.game_types import HasPosition, ResourceType, StructureType


class BaseAction(ABC):
    """Abstract base class for all game actions."""

    @property
    @abstractmethod
    def action_type(self) -> str:
        """The action type identifier."""
        pass

    @abstractmethod
    def to_api_action(self) -> Action:
        """Convert this action to an API action."""
        pass

    @abstractmethod
    def apply(self, game_state: GameState) -> None:
        """Apply this action to the game state."""
        pass


class BuildAction(BaseAction):
    """BUILD action: Build a structure at specified coordinates."""

    def __init__(self, x: int, y: int, structure_type: StructureType):
        self.x = x
        self.y = y
        self.type = structure_type

    @property
    def action_type(self) -> str:
        return "BUILD"

    def to_api_action(self) -> Action:
        from api_client.models import Action

        return Action(action="BUILD", args={"x": self.x, "y": self.y, "type": self.type.value})

    def __str__(self) -> str:
        return f"BUILD {self.type.value} at ({self.x},{self.y})"

    def apply(self, game_state: GameState) -> None:
        """Apply this BUILD action to the game state."""
        # Create the concrete structure to get build costs
        from .structures import BaseStructure, GameStructure, RoadStructure, StoneQuarryStructure

        structure: GameStructure

        if self.type == StructureType.ROAD:
            structure = RoadStructure(x=self.x, y=self.y)
        elif self.type == StructureType.STONE_QUARRY:
            structure = StoneQuarryStructure(x=self.x, y=self.y)
        elif self.type == StructureType.BASE:
            raise ValueError(f"Cannot build type: {self.type}")
            structure = BaseStructure(x=self.x, y=self.y)
        else:
            raise ValueError(f"Unknown structure type: {self.type}")

        # Find base structure using the helper method
        base = game_state.base

        # Deduct costs from base - could go negative, let's accept that - this is for simulation only
        base.storage.subtract_in_place(structure.build_cost)

        # Add the concrete structure to game state
        game_state.add_structure(structure)


class ClaimWinAction(BaseAction):
    """CLAIM_WIN action: Claim victory at specified coordinates."""

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    @property
    def action_type(self) -> str:
        return "CLAIM_WIN"

    def to_api_action(self) -> Action:
        from api_client.models import Action

        return Action(action="CLAIM_WIN", args={"x": self.x, "y": self.y})

    def __str__(self) -> str:
        return f"CLAIM_WIN at ({self.x},{self.y})"

    def apply(self, game_state: GameState) -> None:
        """Apply this CLAIM_WIN action to the game state."""
        # This action doesn't modify the game state, just marks victory


class ExtractAction(BaseAction):
    """MINE action: Extract resources at the structure's rate.

    Notes from rules:
    - Extracts up to the structure's rate per activation.
    - Output stored in the mining structure; transfer to a BaseOnly structure to spend on BUILD.
    - Each structure can be activated at most once per turn.
    """

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    @property
    def action_type(self) -> str:
        return "MINE"

    def to_api_action(self) -> Action:
        from api_client.models import Action

        return Action(action="MINE", args={"x": self.x, "y": self.y})

    def __str__(self) -> str:
        return f"MINE at ({self.x},{self.y})"

    def apply(self, game_state: GameState) -> None:
        """Apply this MINE action to the game state."""
        from .structures import ExtractionStructure, StorageStructure

        structure = game_state.get_structure_at(self.x, self.y)

        assert isinstance(structure, ExtractionStructure), f"Cannot mine on non-extraction structure: {structure}"
        assert isinstance(structure, StorageStructure), f"Cannot mine on non-storage structure: {structure}"

        structure.storage.add(structure.extracted_resource, structure.rate)
        structure.extracted_this_turn = True


class TransferAction(BaseAction):
    """TRANSFER action: Move resources along a path.

    Path constraints per stage1 rules:
    - path is list of [x, y] coordinates.
    - first and last must be Storage structures; intermediates must be Transport structures.
    - length must be at least 3 (source -> transport(s) -> destination).
    """

    def __init__(self, path: list[HasPosition], resource: ResourceType, amount: int):
        self.path = path
        self.resource = resource
        self.amount = amount

    @property
    def action_type(self) -> str:
        return "TRANSFER"

    def to_api_action(self) -> Action:
        from api_client.models import Action

        return Action(
            action="TRANSFER",
            args={"path": [[x, y] for x, y in self.path], "resource": self.resource.value, "amount": self.amount},
        )

    def __str__(self) -> str:
        start, end = self.path[0], self.path[-1]
        return f"TRANSFER {self.amount} {self.resource.value} ({start[0]},{start[1]})→({end[0]},{end[1]})"

    def apply(self, game_state: GameState) -> None:
        """Apply this TRANSFER action to the game state."""
        assert len(self.path) > 1, f"Invalid transfer path: {self.path}"
        assert self.path[0] != self.path[-1], f"Invalid transfer path: {self.path}"

        from game_simulation.structure_mixins import StorageStructure

        source_storage = game_state.get_structure_at(*self.path[0])
        dest_storage = game_state.get_structure_at(*self.path[-1])

        assert isinstance(source_storage, StorageStructure), (
            f"Source storage is not a StorageStructure: {source_storage}"
        )
        assert isinstance(dest_storage, StorageStructure), (
            f"Destination storage is not a StorageStructure: {dest_storage}"
        )

        # Move resources from source to destination
        source_amount = source_storage.storage[self.resource]
        transfer_amount = min(self.amount, source_amount)

        if transfer_amount > 0:
            source_storage.storage.remove(self.resource, transfer_amount)
            dest_storage.storage.add(self.resource, transfer_amount)


AnyAction = BuildAction | ClaimWinAction | ExtractAction | TransferAction

# Mapping used for capability checks and possible factory helpers.
ACTION_TYPE_MAP: dict[str, type[BaseAction]] = {
    "BUILD": BuildAction,
    "CLAIM_WIN": ClaimWinAction,
    "MINE": ExtractAction,
    "TRANSFER": TransferAction,
}


def supported_action_types() -> set[str]:
    """Return action identifiers implemented by the solver."""
    return set(ACTION_TYPE_MAP.keys())
