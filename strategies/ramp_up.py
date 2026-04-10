"""Minimal ramp-up strategy built for quick developer tests."""

from __future__ import annotations

from collections.abc import Generator

from game_simulation.actions import (
    BaseAction,
    BuildAction,
    ClaimWinAction,
    ExtractAction,
    TransferAction,
)
from game_simulation.game_types import ResourceType, StructureType, TerrainType

from strategies.base_strategy import BaseStrategy


class RampUpStrategy(BaseStrategy):
    """Crude strategy that assumes a single mine connected by one planned road segment."""

    def __init__(self, game_state):
        super().__init__(game_state)

        board = self.game_state.board
        base = self.game_state.base

        self._base_pos = (base.x, base.y)

        stone_node = next(node for node in board.resource_nodes if node.resource == "STONE")
        self._quarry_pos = (stone_node.x, stone_node.y)

        planned_road = next(
            (x, y)
            for y, row in enumerate(board.grid)
            for x, tile in enumerate(row)
            if tile == TerrainType.PLANNED_ROAD.value
        )
        self._road_pos = planned_road

        self._transfer_path = [self._quarry_pos, self._road_pos, self._base_pos]

        self._built_road = False
        self._built_quarry = False
        self._claimed = False

    def generate_more_turn_actions(self) -> Generator[BaseAction, None, None]:  # type: ignore[override]
        if self._claimed:
            return

        if not self._built_road:
            self._built_road = True
            yield BuildAction(x=self._road_pos[0], y=self._road_pos[1], structure_type=StructureType.ROAD)

        if not self._built_quarry:
            self._built_quarry = True
            yield BuildAction(x=self._quarry_pos[0], y=self._quarry_pos[1], structure_type=StructureType.STONE_QUARRY)

        structure = self.game_state.get_structure_at(*self._quarry_pos)
        if not structure.extracted_this_turn:
            yield ExtractAction(x=self._quarry_pos[0], y=self._quarry_pos[1])

            amount = structure.storage.get(ResourceType.STONE, 0)
            if amount > 0:
                yield TransferAction(
                    path=self._transfer_path,
                    resource=ResourceType.STONE,
                    amount=amount,
                )

        if self.game_state.base.storage.at_least(self.game_state.goal_resources):
            self._claimed = True
            yield ClaimWinAction(x=self._base_pos[0], y=self._base_pos[1])
