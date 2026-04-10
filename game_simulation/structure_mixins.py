from __future__ import annotations

from game_simulation.actions import BaseAction, ExtractAction
from game_simulation.game_types import ResourceType, StructureType
from game_simulation.inventory import Inventory


class Structure:
    """Base structure model for game simulation."""

    type: StructureType

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __str__(self):
        return f"{self.type.value} at ({self.x},{self.y})"


class BuildableStructure(Structure):
    """Mixin for structures that can be built."""

    build_cost: Inventory

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TransportStructure(Structure):
    """Mixin for structures that can transport resources (path intermediates)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class StorageStructure(Structure):
    """Mixin for structures that can store resources."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage = Inventory()

    def has_resources(self, resources: dict[ResourceType, int] | Inventory) -> bool:
        """Check if this storage structure has the specified resources."""
        if isinstance(resources, dict):
            return self.storage.has_resources(resources)
        return self.storage.at_least(resources)


class ExtractionStructure(StorageStructure):
    """Mixin for structures that can extract resources."""

    _resource_to_structure_type: dict[ResourceType, type[ExtractionStructure]] = {}
    rate: int
    extracted_resource: ResourceType

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extracted_this_turn: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Register the subclass with its extracted resource type
        ExtractionStructure._resource_to_structure_type[cls.extracted_resource] = cls

    @classmethod
    def get_structure_type_for_resource(cls, resource_type: ResourceType) -> type[ExtractionStructure] | None:
        """Get the appropriate extraction structure class for a given resource type."""
        return cls._resource_to_structure_type.get(resource_type)

    def get_extract_actions(self) -> list[BaseAction]:
        return [ExtractAction(self.x, self.y)]

    @property
    def can_be_extracted(self):
        return not self.extracted_this_turn


class BaseOnlyStructure(Structure):
    """Mixin for structures that are base-only (can spend resources for building)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
