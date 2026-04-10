from __future__ import annotations

from abc import ABC, abstractmethod

from api_client.models import LevelDefinition
from game_simulation.game_state import GameState
from game_simulation.game_types import StructureType, TerrainType
from game_simulation.structure_mixins import ExtractionStructure

# Terrain emoji mappings using well-known types
TERRAIN_EMOJIS: dict[str, str] = {
    TerrainType.GRASS: "🟩",
    TerrainType.PLANNED_ROAD: "⬜",
}

# Structure emoji mappings for known structure enums
STRUCTURE_EMOJIS: dict[StructureType, str] = {
    StructureType.BASE: "🏠",
    StructureType.ROAD: "⬛",
}

RESOURCE_NODE_ICON = "◈"
MINE_ICON = "⛏"
OTHER_STRUCTURE_ICON = "⌂"

# Resource emoji mappings using well-known types (kept for backwards compatibility)
RESOURCE_EMOJIS: dict[str, str] = {}


def _first_letter_token(name: str) -> str:
    for ch in name:
        if ch.isalpha():
            return ch.upper()
    return "?"


def _last_letter_token(name: str) -> str:
    for ch in reversed(name):
        if ch.isalpha():
            return ch.upper()
    return "?"


# Fallback for unknown types - use square with representative letter
def _fallback_glyph(name: str) -> str:
    return f"⬜{_first_letter_token(name)}" if name else "⬜?"


def _emoji_for_terrain(terrain: str) -> str:
    """Get emoji for terrain type. Empty/unknown terrain gets empty square."""
    if not terrain:
        return "⬜"
    return TERRAIN_EMOJIS.get(terrain, _fallback_glyph(terrain))


def _resource_symbol(resource: str) -> str:
    """Glyph for a resource node using a generic icon and the first letter."""
    if resource in RESOURCE_EMOJIS:
        return RESOURCE_EMOJIS[resource]
    return f"{RESOURCE_NODE_ICON}{_first_letter_token(resource)}"


def _structure_symbol(structure) -> str:
    """Glyph for a structure with minimal heuristics."""
    struct_type = getattr(structure, "type", None)

    if isinstance(struct_type, StructureType):
        glyph = STRUCTURE_EMOJIS.get(struct_type)
        if glyph:
            return glyph

    if isinstance(structure, ExtractionStructure):
        extracted = getattr(structure, "extracted_resource", None)
        resource_name = getattr(extracted, "value", None)
        if not isinstance(resource_name, str):
            resource_name = str(extracted) if extracted is not None else str(struct_type)
        return f"{MINE_ICON}{_first_letter_token(resource_name)}"

    type_name = getattr(struct_type, "value", None)
    if not isinstance(type_name, str):
        type_name = str(struct_type) if struct_type is not None else structure.__class__.__name__

    return f"{OTHER_STRUCTURE_ICON}{_last_letter_token(type_name)}"


class VisualizationLayer(ABC):
    """Abstract base class for visualization layers."""

    @abstractmethod
    def get_symbol(self, x: int, y: int, state: GameState) -> str | None:
        """
        Get the symbol to display at position (x, y).

        Returns None if this layer has nothing to display at this position.
        """
        pass

    @abstractmethod
    def get_priority(self) -> int:
        """
        Get the priority of this layer. Higher numbers take precedence.
        """
        pass


class TerrainLayer(VisualizationLayer):
    """Layer for displaying terrain/tile types."""

    def get_symbol(self, x: int, y: int, state: GameState) -> str | None:
        if not (0 <= x < state.board.width and 0 <= y < state.board.height):
            return None

        tile = state.board.grid[y][x] if y < len(state.board.grid) and x < len(state.board.grid[y]) else ""
        return _emoji_for_terrain(tile)

    def get_priority(self) -> int:
        return 1


class ResourceLayer(VisualizationLayer):
    """Layer for displaying resource nodes."""

    def get_symbol(self, x: int, y: int, state: GameState) -> str | None:
        for resource_node in state.board.resource_nodes:
            if resource_node.x == x and resource_node.y == y:
                return _resource_symbol(resource_node.resource)
        return None

    def get_priority(self) -> int:
        return 2


class StructureLayer(VisualizationLayer):
    """Layer for displaying structures."""

    def get_symbol(self, x: int, y: int, state: GameState) -> str | None:
        structure = state.get_structure_at(x, y)
        if structure:
            return _structure_symbol(structure)
        return None

    def get_priority(self) -> int:
        return 3


class LayeredRenderer:
    """Renderer that combines multiple visualization layers."""

    def __init__(self, layers: list[VisualizationLayer]):
        # Sort layers by priority (highest first)
        self.layers = sorted(layers, key=lambda layer: layer.get_priority(), reverse=True)

    def render_symbol(self, x: int, y: int, state: GameState) -> str:
        """Get the symbol to display at position (x, y) from the highest priority layer."""
        for layer in self.layers:
            symbol = layer.get_symbol(x, y, state)
            if symbol is not None:
                return symbol

        # Fallback if no layer provides a symbol
        return "⬜"

    def render(self, state: GameState, title: str | None = None) -> str:
        """Render the complete visualization."""
        w = state.board.width
        h = state.board.height

        # Create column header with indexes
        col_header = "  " + "".join(f"{x:2d}" for x in range(w))

        # Render rows top-to-bottom (y=0 at top)
        rows: list[str] = []
        for y in range(h):
            row_chars: list[str] = []
            for x in range(w):
                symbol = self.render_symbol(x, y, state)
                row_chars.append(symbol)
            # Add row index on the left
            rows.append(f"{y:2d}{''.join(row_chars)}")

        # Compose final text with header and board
        if title:
            print(title)
        lines = [col_header, *rows]

        return "\n".join(lines)


def print_progressive_layers(
    state: GameState, layers: list[VisualizationLayer] | None = None, titles: list[str] | None = None
) -> None:
    """Print progressive layer overlays side-by-side."""
    if layers is None:
        layers = [
            TerrainLayer(),
            ResourceLayer(),
            StructureLayer(),
        ]

    if not layers:
        return

    rendered_boards = []
    layer_titles = titles or [f"Layer {i + 1}" for i in range(len(layers))]

    # Generate progressive overlays
    for i in range(len(layers)):
        # Use layers up to index i (inclusive)
        current_layers = layers[: i + 1]
        renderer = LayeredRenderer(current_layers)
        title = layer_titles[i] if i < len(layer_titles) else f"Layer {i + 1}"
        text = renderer.render(state, title)
        rendered_boards.append(text.split("\n"))

    # Print side-by-side
    max_lines = max(len(board) for board in rendered_boards)
    for line_idx in range(max_lines):
        line_parts = []
        for board in rendered_boards:
            if line_idx < len(board):
                line_parts.append(board[line_idx])
            else:
                # Pad with empty space if this board has fewer lines
                line_parts.append(" " * len(board[0]) if board else "")
        print("    ".join(line_parts))


def print_state(*states: GameState, progressive: bool = False) -> None:
    """
    Print one or more game states side-by-side using standard layers.

    Args:
        states: Game states to visualize
        progressive: If True and only one state provided, show progressive layer overlay
    """
    if not states:
        return

    if len(states) == 1:
        if progressive:
            layers = [
                TerrainLayer(),
                ResourceLayer(),
                StructureLayer(),
            ]
            titles = ["Terrain", "+ Resources", "+ Structures"]
            print_progressive_layers(states[0], layers, titles)
            return
        else:
            layers = [
                TerrainLayer(),
                ResourceLayer(),
                StructureLayer(),
            ]
            renderer = LayeredRenderer(layers)
            text = renderer.render(states[0])
            print(text)
            return

    # Render each state to text
    rendered_boards = []
    for state in states:
        layers = [
            TerrainLayer(),
            ResourceLayer(),
            StructureLayer(),
        ]
        renderer = LayeredRenderer(layers)
        text = renderer.render(state)
        rendered_boards.append(text.split("\n"))

    # Print side-by-side
    max_lines = max(len(board) for board in rendered_boards)
    for i in range(max_lines):
        line_parts = []
        for board in rendered_boards:
            if i < len(board):
                line_parts.append(board[i])
            else:
                # Pad with empty space if this board has fewer lines
                line_parts.append(" " * len(board[0]) if board else "")
        print("  ".join(line_parts))


def print_level(*levels: LevelDefinition) -> None:
    """Print one or more level definitions side-by-side using standard layers."""
    if not levels:
        return

    if len(levels) == 1:
        print_state(GameState(levels[0]))
        return

    # Render each level to text
    rendered_boards = []
    for level in levels:
        layers = [
            TerrainLayer(),
            ResourceLayer(),
            StructureLayer(),
        ]
        renderer = LayeredRenderer(layers)
        state = GameState(level)
        text = renderer.render(state)
        rendered_boards.append(text.split("\n"))

    # Print side-by-side
    max_lines = max(len(board) for board in rendered_boards)
    for i in range(max_lines):
        line_parts = []
        for board in rendered_boards:
            if i < len(board):
                line_parts.append(board[i])
            else:
                # Pad with empty space if this board has fewer lines
                line_parts.append(" " * len(board[0]) if board else "")
        print("  ".join(line_parts))


# Legacy compatibility functions (keep existing API)
def parse_state(state: GameState) -> str:
    """Legacy function for backward compatibility."""
    layers = [
        TerrainLayer(),
        ResourceLayer(),
        StructureLayer(),
    ]
    renderer = LayeredRenderer(layers)
    return renderer.render(state)
