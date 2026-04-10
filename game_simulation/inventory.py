"""Inventory model for managing resource collections in game simulation."""

from __future__ import annotations

from collections.abc import Iterator

from mypy.types import Iterable

from game_simulation.game_types import ResourceType


class Inventory:
    """A collection of resources with convenience methods for game operations."""

    def __init__(self, resources: dict[ResourceType, int] | None = None):
        """Initialize inventory with optional starting resources."""
        self._resources: dict[ResourceType, int] = resources.copy() if resources else {}

    def __getitem__(self, resource: ResourceType) -> int:
        """Get amount of a resource, returns 0 if not present."""
        return self._resources.get(resource, 0)

    def __setitem__(self, resource: ResourceType, amount: int) -> None:
        """Set amount of a resource."""
        if amount <= 0:
            self._resources.pop(resource, None)
        else:
            self._resources[resource] = amount

    def __contains__(self, resource: ResourceType) -> bool:
        """Check if resource exists in inventory with amount > 0."""
        return self._resources.get(resource, 0) > 0

    def __iter__(self) -> Iterator[tuple[ResourceType, int]]:
        """Iterate over resource-amount pairs."""
        return iter(self._resources.items())

    def __len__(self) -> int:
        """Number of different resource types in inventory."""
        return len(self._resources)

    def __bool__(self) -> bool:
        """True if inventory has any resources with amount > 0."""
        return any(amount > 0 for amount in self._resources.values())

    def __repr__(self) -> str:
        return f"Inventory({dict(self._resources)})"

    def __add__(self, other: Inventory) -> Inventory:
        """Add two inventories together."""
        return self.merge(other)

    def get(self, resource: ResourceType, default: int = 0) -> int:
        """Get amount of a resource with default value."""
        return self._resources.get(resource, default)

    def add(self, resource: ResourceType, amount: int) -> None:
        """Add resources to inventory."""
        if amount > 0:
            self[resource] = self[resource] + amount

    def remove(self, resource: ResourceType, amount: int) -> None:
        """Remove resources from inventory. Amount can go negative."""
        self[resource] = self[resource] - amount

    def at_least(self, other: Inventory) -> bool:
        """Check if this inventory contains at least the amounts in other inventory."""
        return all(self[resource] >= amount for resource, amount in other)

    def has_resources(self, resources: dict[ResourceType, int]) -> bool:
        """Check if this inventory has at least the specified resources."""
        return all(self[resource] >= amount for resource, amount in resources.items())

    def merge(self, other: Inventory) -> Inventory:
        """Create new inventory by merging this with other (adding amounts)."""
        result = Inventory(self._resources)
        for resource, amount in other:
            result.add(resource, amount)
        return result

    def merge_in_place(self, other: Inventory) -> None:
        """Add all resources from other inventory into this one."""
        for resource, amount in other:
            self.add(resource, amount)

    def subtract(self, other: Inventory) -> Inventory:
        """Create new inventory by subtracting other from this."""
        result = Inventory(self._resources)
        for resource, amount in other:
            result.remove(resource, amount)
        return result

    def subtract_in_place(self, other: Inventory) -> None:
        """Subtract all resources from other inventory from this one."""
        for resource, amount in other:
            self.remove(resource, amount)

    def multiply(self, factor: int) -> Inventory:
        """Create new inventory with all amounts multiplied by factor."""
        return Inventory({resource: amount * factor for resource, amount in self})

    def copy(self) -> Inventory:
        """Create a copy of this inventory."""
        return Inventory(self._resources)

    def clear(self) -> None:
        """Remove all resources from inventory."""
        self._resources.clear()

    def to_dict(self) -> dict[ResourceType, int]:
        """Convert to dictionary representation."""
        return self._resources.copy()

    @classmethod
    def from_dict(cls, resources: dict[ResourceType, int]) -> Inventory:
        """Create inventory from dictionary."""
        return cls(resources)

    @classmethod
    def total(cls, inventories: Iterable[Inventory]) -> Inventory:
        """Create new inventory with total amount of all inventories."""
        total = Inventory()
        for inv in inventories:
            total += inv
        return total

    def types(self) -> set[ResourceType]:
        return set(self._resources.keys())

    def missing_to(self, target: Inventory) -> Inventory:
        """Calculate missing resources needed to reach target inventory.

        Returns only resources where this inventory has less than target.
        Resources where this inventory has equal or more are not included.
        """
        missing = Inventory()
        for resource, target_amount in target:
            current_amount = self[resource]
            if current_amount < target_amount:
                missing[resource] = target_amount - current_amount
        return missing
