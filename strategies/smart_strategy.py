"""Smart strategy that finds the nearest stone node and mines enough to reach the goal."""

from __future__ import annotations

import math
from collections import deque
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


def find_path_to_nearest_stone(game_state):
    base = game_state.base
    base_pos = (base.x, base.y)
    board = game_state.board
    width = board.width
    height = board.height
    grid = board.grid

    stone_positions = {
        (node.x, node.y)
        for node in board.resource_nodes
        if node.resource == "STONE"
    }

    def is_valid(x, y, visited):
        if not (0 <= x < width and 0 <= y < height):
            return False
        if (x, y) in visited:
            return False
        tile = grid[y][x]
        is_buildable = tile in (TerrainType.GRASS.value, TerrainType.PLANNED_ROAD.value)
        is_destination = (x, y) in stone_positions
        return is_buildable or is_destination

    queue = deque()
    queue.append(base_pos)
    visited = {base_pos}
    parent = {base_pos: None}

    while queue:
        x, y = queue.popleft()

        if (x, y) in stone_positions:
            path = []
            current = (x, y)
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path, (x, y)

        for nx, ny in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if is_valid(nx, ny, visited):
                visited.add((nx, ny))
                parent[(nx, ny)] = (x, y)
                queue.append((nx, ny))

    return None, None


class SmartStrategy(BaseStrategy):

    def __init__(self, game_state):
        super().__init__(game_state)

        # Step 1: find path to nearest stone node
        path, quarry_pos = find_path_to_nearest_stone(game_state)

        self._base_pos = (game_state.base.x, game_state.base.y)
        self._quarry_pos = quarry_pos

        # roads are everything between BASE and quarry
        self._roads_to_build = path[1:-1]

        # transfer path: quarry -> roads -> base
        self._transfer_path = list(reversed(path))

        # Step 2: calculate how many times we need to mine
        road_cost = len(self._roads_to_build) * 1
        quarry_cost = 10
        total_build_cost = road_cost + quarry_cost

        current_stone = game_state.base.storage.get(ResourceType.STONE, 0)
        stones_after_build = current_stone - total_build_cost

        target = game_state.goal_resources.get(ResourceType.STONE, 0)
        stones_needed = max(0, target - stones_after_build)

        self._mines_needed = math.ceil(stones_needed / 5)
        self._mines_done = 0
        self._built = False
        self._last_turn_acted = -1

        print(f"Base pos: {self._base_pos}")
        print(f"Quarry pos: {self._quarry_pos}")
        print(f"Roads to build: {self._roads_to_build}")
        print(f"Road cost: {road_cost}, Quarry cost: {quarry_cost}")
        print(f"Starting stone: {current_stone}")
        print(f"Stone after build: {stones_after_build}")
        print(f"Target: {target}")
        print(f"Mines needed: {self._mines_needed}")
        print(f"Total turns needed: {1 + self._mines_needed} (1 build + {self._mines_needed} mines)")
        print(f"Max turns allowed: {game_state.max_turns}")

    def generate_more_turn_actions(self) -> Generator[BaseAction, None, None]:

        current_turn = self.game_state.executed_turns
        # print(f"Turn {current_turn}: built={self._built}, mines_done={self._mines_done}, mines_needed={self._mines_needed}, last_acted={self._last_turn_acted}")
        if self._last_turn_acted == current_turn:
            return  # already acted this turn, yield nothing
        
        self._last_turn_acted = current_turn

        # Phase 1: build (only on first turn)
        if not self._built:
            self._built = True
            for (rx, ry) in self._roads_to_build:
                yield BuildAction(x=rx, y=ry, structure_type=StructureType.ROAD)
            yield BuildAction(
                x=self._quarry_pos[0],
                y=self._quarry_pos[1],
                structure_type=StructureType.STONE_QUARRY
            )

        # Phase 2: mine + transfer (one mine per turn)
        elif self._mines_done < self._mines_needed:
            self._mines_done += 1
            yield ExtractAction(x=self._quarry_pos[0], y=self._quarry_pos[1])
            quarry = self.game_state.get_structure_at(*self._quarry_pos)
            amount = quarry.storage.get(ResourceType.STONE, 0)
            yield TransferAction(
                path=self._transfer_path,
                resource=ResourceType.STONE,
                amount=amount,
            )
            
            # If this was the last mine, claim win in the same turn!
            if self._mines_done == self._mines_needed:
                yield ClaimWinAction(x=self._base_pos[0], y=self._base_pos[1])