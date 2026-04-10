from __future__ import annotations

from enum import Enum
from typing import NamedTuple, Protocol


class ResourceType(str, Enum):
    STONE = "STONE"


class StructureType(str, Enum):
    ROAD = "ROAD"
    STONE_QUARRY = "STONE_QUARRY"
    BASE = "BASE"


class TerrainType(str, Enum):
    GRASS = "GRASS"
    PLANNED_ROAD = "PLANNED_ROAD"


class StructureInterface(str, Enum):
    Buildable = "Buildable"
    Transport = "Transport"
    Storage = "Storage"
    Extraction = "Extraction"
    BaseOnly = "BaseOnly"


class HasPosition(Protocol):
    """Protocol for objects that have x and y attributes."""

    x: int
    y: int


class Position(NamedTuple):
    """A position in the game board."""

    x: int
    y: int


def supported_resource_types() -> set[str]:
    """Return resource identifiers implemented by the solver."""
    return {resource.value for resource in ResourceType}
