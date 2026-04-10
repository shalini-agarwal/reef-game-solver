"""Inventory visualization functions for the game simulation."""

from __future__ import annotations

from game_simulation.game_types import ResourceType
from game_simulation.inventory import Inventory


def _get_resource_name(resource_type: ResourceType) -> str:
    """Get a human-readable name for the resource type."""
    return resource_type.value.replace("_", " ").title()


def print_inventory(inventory: Inventory, title: str | None = None) -> None:
    """
    Print inventory as a nicely formatted table with resource names and counts.

    Args:
        inventory: The inventory to display
        title: Optional title to display above the table
    """
    if title:
        print(title)

    if not inventory:
        print("  (empty)")
        return

    # Find the longest resource name for formatting
    max_name_length = max(len(_get_resource_name(resource)) for resource, _ in inventory)
    max_count_length = max(len(str(amount)) for _, amount in inventory)

    # Print header
    print(f"  {'Resource':<{max_name_length}} {'Count':>{max_count_length}}")
    print(f"  {'-' * max_name_length} {'-' * max_count_length}")

    # Print each resource
    for resource, amount in sorted(inventory, key=lambda x: x[0].value):
        name = _get_resource_name(resource)
        print(f"  {name:<{max_name_length}} {amount:>{max_count_length}}")


def format_inventory_multiline(inventory: Inventory) -> str:
    """Return a multi-line string describing inventory contents resource-by-resource."""

    if not inventory:
        return "(empty)"

    lines: list[str] = []

    def _sort_key(resource: ResourceType | str) -> str:
        return resource.value if isinstance(resource, ResourceType) else str(resource)

    def _display_name(resource: ResourceType | str) -> str:
        if isinstance(resource, ResourceType):
            return _get_resource_name(resource)
        return str(resource).replace("_", " ").title()

    for resource, amount in sorted(inventory, key=lambda x: _sort_key(x[0])):
        name = _display_name(resource)
        lines.append(f"{name}: {amount}")

    return "\n".join(lines)


def format_win_condition(target_resources: dict[str, int], padding: int = 0) -> str:
    """
    Format win condition as an inline inventory representation.

    Args:
        target_resources: Dictionary of resource string to amount
        padding: Number of spaces to pad the result to (for alignment)

    Returns:
        String representation like "Stone: 10 Iron: 5" or padded version
    """
    if not target_resources:
        result = "(no requirements)"
    else:
        parts = []
        for resource_str, amount in sorted(target_resources.items()):
            try:
                resource_type = ResourceType(resource_str)
                name = _get_resource_name(resource_type)
            except ValueError:
                # Unknown resource type, use fallback name formatting
                name = resource_str.replace("_", " ").title()
            parts.append(f"{name}: {amount}")
        result = " ".join(parts)

    # Apply padding if requested
    if padding > 0:
        result = result.ljust(padding)

    return result
