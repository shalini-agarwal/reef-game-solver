"""Smart strategy using 0-1 BFS for optimal routing, mine-first turn order."""

from __future__ import annotations

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
from strategies.base_strategy import BaseStrategy, StrategyFailed


def find_path_to_nearest_stone(
    game_state,
    already_targeted=None,
    network_positions=None,
    quarry_positions=None,
):
    """
    Find path from BASE to nearest untargeted stone node using 0-1 BFS.
    Cost = number of NEW roads needed.
    Quarry positions cannot be used as intermediates (only roads can).
    Always reconstructs full path from BASE so transfer paths are always valid.
    """
    if already_targeted is None:
        already_targeted = set()
    if network_positions is None:
        network_positions = {(game_state.base.x, game_state.base.y)}
    if quarry_positions is None:
        quarry_positions = set()

    board = game_state.board
    width = board.width
    height = board.height
    grid = board.grid
    base_pos = (game_state.base.x, game_state.base.y)

    stone_positions = {
        (node.x, node.y)
        for node in board.resource_nodes
        if node.resource == "STONE"
        and (node.x, node.y) not in already_targeted
    }

    if not stone_positions:
        return None, None

    def in_bounds(x, y):
        return 0 <= x < width and 0 <= y < height

    def is_traversable(x, y):
        if (x, y) in stone_positions:
            return True
        if (x, y) in quarry_positions:
            return False
        if (x, y) in network_positions:
            return True
        tile = grid[y][x]
        return tile in (TerrainType.GRASS.value, TerrainType.PLANNED_ROAD.value)

    dist = {base_pos: 0}
    parent = {base_pos: None}
    queue = deque()
    queue.append((0, base_pos))

    best_stone_node = None

    while queue:
        d, (x, y) = queue.popleft()

        if d > dist.get((x, y), float('inf')):
            continue

        if (x, y) in stone_positions:
            best_stone_node = (x, y)
            break

        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
            if not in_bounds(nx, ny):
                continue
            if not is_traversable(nx, ny):
                continue

            tile_cost = 0 if (nx, ny) in network_positions else 1
            new_dist = d + tile_cost

            if new_dist < dist.get((nx, ny), float('inf')):
                dist[(nx, ny)] = new_dist
                parent[(nx, ny)] = (x, y)
                if tile_cost == 0:
                    queue.appendleft((new_dist, (nx, ny)))
                else:
                    queue.append((new_dist, (nx, ny)))

    if best_stone_node is None:
        return None, None

    path = []
    current = best_stone_node
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()

    return path, best_stone_node


def find_all_paths(game_state):
    """
    Find paths to all stone nodes using 0-1 BFS, extending network each time.
    Tracks quarry positions separately so they are never used as intermediates.
    """
    plans = []
    already_targeted = set()
    network_positions = set(game_state.structures.keys())
    quarry_positions = set()

    while True:
        path, quarry_pos = find_path_to_nearest_stone(
            game_state,
            already_targeted,
            network_positions,
            quarry_positions,
        )
        if path is None:
            break

        new_roads = [pos for pos in path[1:-1] if pos not in network_positions]
        road_cost = len(new_roads)
        total_cost = road_cost + 10

        plans.append((path, quarry_pos, total_cost, new_roads))
        already_targeted.add(quarry_pos)

        # roads go into network (traversable as intermediates)
        for pos in path[1:-1]:
            network_positions.add(pos)

        # quarry goes into both sets
        quarry_positions.add(quarry_pos)
        network_positions.add(quarry_pos)

    return plans


def simulate_plan(starting_stone, target, max_turns, quarry_costs):
    """
    Simulate mine-first turn order:
      1. Mine all existing quarries → transfer to base
      2. Build as many quarries as affordable (each newly built quarry also mines immediately)
      3. Check if target reached

    This matches generate_more_turn_actions turn order exactly.
    Returns list of (turn, quarry_index) build events if reachable, else None.
    """
    stone = starting_stone
    built_quarries = 0
    next_to_build = 0
    build_events = []

    for turn in range(max_turns):
        # Step 1: mine all existing quarries
        stone += built_quarries * 5

        # Step 2: build as many quarries as affordable,
        # each newly built one also mines immediately in this turn
        while next_to_build < len(quarry_costs):
            cost = quarry_costs[next_to_build]
            if stone >= cost:
                stone -= cost
                build_events.append((turn, next_to_build))
                built_quarries += 1
                next_to_build += 1
                stone += 5  # new quarry mines immediately same turn
            else:
                break

        # Step 3: check goal
        if stone >= target:
            return build_events

    return None


def find_optimal_quarry_subset(quarry_plans, starting_stone, target, max_turns):
    """
    Try increasing subsets of quarries (in order found by BFS) until we find
    the minimum number that reaches the target within max_turns.
    """
    for n in range(1, len(quarry_plans) + 1):
        subset = quarry_plans[:n]
        subset_costs = [cost for (_, _, cost, _) in subset]
        build_schedule = simulate_plan(starting_stone, target, max_turns, subset_costs)
        if build_schedule is not None:
            print(f"Optimal plan: {n} quarries needed")
            print(f"Build schedule: {build_schedule}")
            return subset, build_schedule

    return None, None


class SmartStrategy(BaseStrategy):

    def __init__(self, game_state):
        super().__init__(game_state)

        self._base_pos = (game_state.base.x, game_state.base.y)

        # Step 1: find paths to all stone nodes
        all_quarry_plans = find_all_paths(game_state)

        starting_stone = game_state.base.storage.get(ResourceType.STONE, 0)
        target = game_state.goal_resources.get(ResourceType.STONE, 0)
        max_turns = game_state.max_turns

        print(f"Found {len(all_quarry_plans)} quarry plans:")
        for i, (path, qpos, cost, new_roads) in enumerate(all_quarry_plans):
            print(f"  Quarry {i+1}: pos={qpos}, cost={cost}, new_roads={new_roads}")
        print(f"Starting stone: {starting_stone}, Target: {target}, Max turns: {max_turns}")

        # Step 2: find minimum quarries needed
        selected_plans, build_schedule = find_optimal_quarry_subset(
            all_quarry_plans, starting_stone, target, max_turns
        )

        if selected_plans is None:
            raise StrategyFailed(
                "Cannot reach target within turn limit with any number of quarries."
            )

        # Step 3: store plan for execution
        self._selected_plans = selected_plans
        self._build_schedule = build_schedule
        self._built_quarries = []
        self._transfer_paths = {}
        self._next_to_build = 0       # index into selected_plans
        self._last_turn_acted = -1

    def generate_more_turn_actions(self) -> Generator[BaseAction, None, None]:

        current_turn = self.game_state.executed_turns
        if self._last_turn_acted == current_turn:
            return
        self._last_turn_acted = current_turn

        # Step 1: mine ALL existing quarries first
        for qpos in self._built_quarries:
            yield ExtractAction(x=qpos[0], y=qpos[1])

        # Step 2: transfer ALL existing quarries to BASE
        for qpos in self._built_quarries:
            quarry = self.game_state.get_structure_at(*qpos)
            amount = quarry.storage.get(ResourceType.STONE, 0)
            if amount > 0:
                yield TransferAction(
                    path=self._transfer_paths[qpos],
                    resource=ResourceType.STONE,
                    amount=amount,
                )

        # Step 3: build as many new quarries as we can now afford
        # (using stone just transferred from existing quarries)
        while self._next_to_build < len(self._selected_plans):
            current_stone = self.game_state.base.storage.get(ResourceType.STONE, 0)
            path, quarry_pos, cost, new_roads = self._selected_plans[self._next_to_build]

            if current_stone < cost:
                break  # can't afford next quarry this turn

            # build new roads
            for (rx, ry) in new_roads:
                yield BuildAction(x=rx, y=ry, structure_type=StructureType.ROAD)

            # build quarry
            yield BuildAction(
                x=quarry_pos[0],
                y=quarry_pos[1],
                structure_type=StructureType.STONE_QUARRY
            )

            # store transfer path and register quarry
            self._transfer_paths[quarry_pos] = list(reversed(path))
            self._built_quarries.append(quarry_pos)
            self._next_to_build += 1

            # Step 4: immediately mine and transfer the new quarry in this same turn
            yield ExtractAction(x=quarry_pos[0], y=quarry_pos[1])
            quarry = self.game_state.get_structure_at(*quarry_pos)
            amount = quarry.storage.get(ResourceType.STONE, 0)
            if amount > 0:
                yield TransferAction(
                    path=self._transfer_paths[quarry_pos],
                    resource=ResourceType.STONE,
                    amount=amount,
                )

        # Step 5: claim win if goal reached
        if self.game_state.base.storage.at_least(self.game_state.goal_resources):
            yield ClaimWinAction(x=self._base_pos[0], y=self._base_pos[1])