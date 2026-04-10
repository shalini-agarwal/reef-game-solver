from __future__ import annotations

from typing import TypeAlias

from api_client.models import Structure as ApiStructure

# Import enums from the shared game_types module
from .game_types import ResourceType, StructureType
from .inventory import Inventory
from .structure_mixins import (
    BaseOnlyStructure,
    BuildableStructure,
    ExtractionStructure,
    StorageStructure,
    TransportStructure,
)


class RoadStructure(BuildableStructure, TransportStructure):
    """ROAD structure for game simulation."""

    build_cost = Inventory({ResourceType.STONE: 1})
    type = StructureType.ROAD

    def __init__(self, x: int, y: int):
        super().__init__(x, y)


class StoneQuarryStructure(ExtractionStructure, BuildableStructure):
    """STONE_QUARRY structure for game simulation."""

    build_cost = Inventory({ResourceType.STONE: 10})
    rate = 5
    extracted_resource = ResourceType.STONE
    type = StructureType.STONE_QUARRY

    def __init__(self, x: int, y: int):
        super().__init__(x, y)


class BaseStructure(StorageStructure, BaseOnlyStructure):
    """BASE structure for game simulation."""

    type = StructureType.BASE

    def __init__(self, x: int, y: int):
        super().__init__(x, y)


# Union of all structure types for game simulation
GameStructure: TypeAlias = RoadStructure | StoneQuarryStructure | BaseStructure


def create_structure_from_api(api_structure: ApiStructure, structure_type: StructureType) -> GameStructure:
    """Factory function to create game structure from API structure data."""
    base_data = {
        "x": api_structure.x,
        "y": api_structure.y,
    }

    if structure_type == StructureType.ROAD:
        return RoadStructure(**base_data)
    elif structure_type == StructureType.STONE_QUARRY:
        structure = StoneQuarryStructure(**base_data)
        if api_structure.storage:
            # Update existing storage with API data, don't overwrite
            for k, v in api_structure.storage.items():
                structure.storage[ResourceType(k)] = v
        return structure
    elif structure_type == StructureType.BASE:
        structure = BaseStructure(**base_data)
        if api_structure.storage:
            # Update existing storage with API data, don't overwrite
            for k, v in api_structure.storage.items():
                structure.storage[ResourceType(k)] = v
        return structure
    else:
        raise ValueError(f"Unknown structure type: {structure_type}")


def supported_structure_types() -> set[str]:
    """Return structure identifiers implemented by the solver."""
    return {
        RoadStructure.type.value,
        StoneQuarryStructure.type.value,
        BaseStructure.type.value,
    }
