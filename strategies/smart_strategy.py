"""Smart strategy for Stage 2: handles both STONE and IRON_ORE resources."""

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


def _bfs_to_nearest(
    game_state,
    target_positions,
    already_targeted,
    network_positions,
    extraction_positions,
):
    """
    0-1 BFS from BASE to nearest target position.
    Cost 0 = traverse existing network tile.
    Cost 1 = build new road on grass/planned_road tile.
    Extraction positions blocked as intermediates.
    Always reconstructs full path from BASE.
    """
    board = game_state.board
    width = board.width
    height = board.height
    grid = board.grid
    base_pos = (game_state.base.x, game_state.base.y)

    reachable_targets = target_positions - already_targeted
    if not reachable_targets:
        return None, None

    def in_bounds(x, y):
        return 0 <= x < width and 0 <= y < height

    def is_traversable(x, y):
        if (x, y) in reachable_targets:
            return True
        if (x, y) in extraction_positions:
            return False
        if (x, y) in network_positions:
            return True
        tile = grid[y][x]
        return tile in (TerrainType.GRASS.value, TerrainType.PLANNED_ROAD.value)

    dist = {base_pos: 0}
    parent = {base_pos: None}
    queue = deque([(0, base_pos)])
    best = None

    while queue:
        d, (x, y) = queue.popleft()

        if d > dist.get((x, y), float('inf')):
            continue

        if (x, y) in reachable_targets:
            best = (x, y)
            break

        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
            if not in_bounds(nx, ny) or not is_traversable(nx, ny):
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

    if best is None:
        return None, None

    path = []
    current = best
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path, best


def find_all_paths(game_state):
    """
    Find paths to all resource nodes using 0-1 BFS.
    Stone and iron paths are found INDEPENDENTLY from BASE.
    This ensures iron mine costs are never underestimated.
    We still share roads at EXECUTION time (skipping already-built tiles),
    but costs are calculated conservatively.
    """
    base_network = set(game_state.structures.keys())  # just BASE at start

    stone_plans = []
    iron_plans = []

    # --- find stone quarry paths (extending network as we go) ---
    stone_network = set(base_network)
    stone_extractions = set()
    stone_already_targeted = set()

    stone_nodes = {
        (node.x, node.y)
        for node in game_state.board.resource_nodes
        if node.resource == ResourceType.STONE.value
    }

    while stone_nodes:
        path, pos = _bfs_to_nearest(
            game_state,
            target_positions=stone_nodes,
            already_targeted=stone_already_targeted,
            network_positions=stone_network,
            extraction_positions=stone_extractions,
        )
        if path is None:
            break

        new_roads = [p for p in path[1:-1] if p not in stone_network]
        total_cost = len(new_roads) + 10

        stone_plans.append({
            'path': path,
            'pos': pos,
            'cost': total_cost,
            'new_roads': new_roads,
            'structure_type': StructureType.STONE_QUARRY,
            'resource_type': ResourceType.STONE,
        })

        stone_already_targeted.add(pos)
        for p in path[1:-1]:
            stone_network.add(p)
        stone_extractions.add(pos)
        stone_network.add(pos)
        stone_nodes.discard(pos)

    # --- find iron mine paths INDEPENDENTLY from BASE only ---
    # We do NOT assume any stone quarry roads exist.
    # This ensures costs are never underestimated.
    iron_network = set(base_network)   # ← reset to just BASE
    iron_extractions = set()
    iron_already_targeted = set()

    iron_nodes = {
        (node.x, node.y)
        for node in game_state.board.resource_nodes
        if node.resource == ResourceType.IRON_ORE.value
    }

    while iron_nodes:
        path, pos = _bfs_to_nearest(
            game_state,
            target_positions=iron_nodes,
            already_targeted=iron_already_targeted,
            network_positions=iron_network,
            extraction_positions=iron_extractions,
        )
        if path is None:
            break

        new_roads = [p for p in path[1:-1] if p not in iron_network]
        total_cost = len(new_roads) + 15

        iron_plans.append({
            'path': path,
            'pos': pos,
            'cost': total_cost,
            'new_roads': new_roads,
            'structure_type': StructureType.IRON_MINE,
            'resource_type': ResourceType.IRON_ORE,
        })

        iron_already_targeted.add(pos)
        for p in path[1:-1]:
            iron_network.add(p)
        iron_extractions.add(pos)
        iron_network.add(pos)
        iron_nodes.discard(pos)

    return {'stone': stone_plans, 'iron': iron_plans}


def simulate_plan(starting_stone, iron_target, max_turns, stone_costs, mine_costs):
    """
    Simulate combined stone + iron production.
    Turn order (must match generate_more_turn_actions exactly):
      1. Mine all existing quarries → stone
      2. Mine all existing mines → iron
      3. Build as many structures as affordable (quarries first, then mines)
         each newly built structure also mines immediately
      4. Check if iron target reached
    Returns list of (turn, resource_type, plan_index) or None.
    """
    stone = starting_stone
    iron = 0
    built_quarries = 0
    built_mines = 0
    next_quarry = 0
    next_mine = 0
    build_events = []

    for turn in range(max_turns):
        # step 1: mine existing structures
        stone += built_quarries * 5
        iron += built_mines * 5

        # step 2: build as many as affordable, keep trying until nothing more fits
        changed = True
        while changed:
            changed = False
            # try next quarry
            if next_quarry < len(stone_costs) and stone >= stone_costs[next_quarry]:
                stone -= stone_costs[next_quarry]
                build_events.append((turn, ResourceType.STONE, next_quarry))
                built_quarries += 1
                next_quarry += 1
                stone += 5  # mines immediately
                changed = True
            # try next iron mine
            if next_mine < len(mine_costs) and stone >= mine_costs[next_mine]:
                stone -= mine_costs[next_mine]
                build_events.append((turn, ResourceType.IRON_ORE, next_mine))
                built_mines += 1
                next_mine += 1
                iron += 5   # mines immediately
                changed = True

        # step 3: check win condition
        if iron >= iron_target:
            return build_events

    return None


def simulate_stone_only(starting_stone, stone_target, max_turns, quarry_costs):
    """Simulate stone-only production (stage 1 fallback)."""
    stone = starting_stone
    built = 0
    next_q = 0
    events = []

    for turn in range(max_turns):
        stone += built * 5
        changed = True
        while changed:
            changed = False
            if next_q < len(quarry_costs) and stone >= quarry_costs[next_q]:
                stone -= quarry_costs[next_q]
                events.append((turn, ResourceType.STONE, next_q))
                built += 1
                next_q += 1
                stone += 5
                changed = True
        if stone >= stone_target:
            return events

    return None


def find_optimal_plan(all_paths, starting_stone, goal_resource, goal_amount, max_turns):
    """
    Find minimum quarries + mines needed to reach goal within max_turns.
    For IRON_ORE goals: tries combinations of stone quarries and iron mines.
    For STONE goals: tries increasing numbers of stone quarries only.
    Returns (selected_stone, selected_iron, build_schedule).
    """
    stone_plans = all_paths['stone']
    iron_plans = all_paths['iron']

    if goal_resource == ResourceType.STONE:
        # stage 1 fallback: stone-only
        for n in range(1, len(stone_plans) + 1):
            costs = [p['cost'] for p in stone_plans[:n]]
            schedule = simulate_stone_only(starting_stone, goal_amount, max_turns, costs)
            if schedule is not None:
                print(f"Stone-only plan: {n} quarries")
                print(f"Build schedule: {schedule}")
                return stone_plans[:n], [], schedule
        return None, None, None

    # stage 2+: iron ore goal
    # try increasing numbers of iron mines, with increasing quarries to fund them
    for num_iron in range(1, len(iron_plans) + 1):
        for num_stone in range(0, len(stone_plans) + 1):
            stone_costs = [p['cost'] for p in stone_plans[:num_stone]]
            mine_costs = [p['cost'] for p in iron_plans[:num_iron]]
            schedule = simulate_plan(
                starting_stone, goal_amount, max_turns, stone_costs, mine_costs
            )
            if schedule is not None:
                print(f"Optimal: {num_stone} quarries + {num_iron} iron mines")
                print(f"Build schedule: {schedule}")
                return stone_plans[:num_stone], iron_plans[:num_iron], schedule

    return None, None, None


class SmartStrategy(BaseStrategy):

    def __init__(self, game_state):
        super().__init__(game_state)

        self._base_pos = (game_state.base.x, game_state.base.y)

        # determine goal resource and amount
        goal = game_state.goal_resources
        if goal.get(ResourceType.IRON_ORE, 0) > 0:
            self._goal_resource = ResourceType.IRON_ORE
            goal_amount = goal.get(ResourceType.IRON_ORE, 0)
        else:
            self._goal_resource = ResourceType.STONE
            goal_amount = goal.get(ResourceType.STONE, 0)

        starting_stone = game_state.base.storage.get(ResourceType.STONE, 0)
        max_turns = game_state.max_turns

        # find all paths
        all_paths = find_all_paths(game_state)

        print(f"Stone plans: {len(all_paths['stone'])}, Iron plans: {len(all_paths['iron'])}")
        for p in all_paths['stone']:
            print(f"  Stone quarry at {p['pos']}, cost={p['cost']}, new_roads={p['new_roads']}")
        for p in all_paths['iron']:
            print(f"  Iron mine at {p['pos']}, cost={p['cost']}, new_roads={p['new_roads']}")
        print(f"Starting stone: {starting_stone}, Goal: {goal_amount} {self._goal_resource.value}, Max turns: {max_turns}")

        # find optimal plan
        selected_stone, selected_iron, build_schedule = find_optimal_plan(
            all_paths, starting_stone, self._goal_resource, goal_amount, max_turns
        )

        if build_schedule is None:
            raise StrategyFailed("Cannot reach target within turn limit.")

        self._selected_stone = selected_stone
        self._selected_iron = selected_iron
        self._build_schedule = build_schedule
        self._built_extractors = []    # list of (pos, resource_type)
        self._transfer_paths = {}      # pos -> full transfer path to BASE
        self._last_turn_acted = -1

    def generate_more_turn_actions(self) -> Generator[BaseAction, None, None]:

        current_turn = self.game_state.executed_turns
        if self._last_turn_acted == current_turn:
            return
        self._last_turn_acted = current_turn

        # step 1: mine all existing extractors
        for (qpos, _) in self._built_extractors:
            yield ExtractAction(x=qpos[0], y=qpos[1])

        # step 2: transfer all to BASE
        for (qpos, resource_type) in self._built_extractors:
            extractor = self.game_state.get_structure_at(*qpos)
            amount = extractor.storage.get(resource_type, 0)
            if amount > 0:
                yield TransferAction(
                    path=self._transfer_paths[qpos],
                    resource=resource_type,
                    amount=amount,
                )

        # step 3: build scheduled structures for this turn
        for (build_turn, resource_type, plan_idx) in self._build_schedule:
            if build_turn != current_turn:
                continue

            plan = (
                self._selected_stone[plan_idx]
                if resource_type == ResourceType.STONE
                else self._selected_iron[plan_idx]
            )

            # KEY FIX: build ALL tiles in full path order, skipping already-built tiles
            # This ensures adjacency is always satisfied regardless of network routing
            for pos in plan['path'][1:-1]:
                if self.game_state.get_structure_at(*pos) is None:
                    yield BuildAction(
                        x=pos[0], y=pos[1],
                        structure_type=StructureType.ROAD
                    )

            # build the extraction structure
            yield BuildAction(
                x=plan['pos'][0],
                y=plan['pos'][1],
                structure_type=plan['structure_type'],
            )

            # store transfer path and register extractor
            self._transfer_paths[plan['pos']] = list(reversed(plan['path']))
            self._built_extractors.append((plan['pos'], plan['resource_type']))

            # mine and transfer new extractor immediately in same turn
            yield ExtractAction(x=plan['pos'][0], y=plan['pos'][1])
            extractor = self.game_state.get_structure_at(*plan['pos'])
            amount = extractor.storage.get(plan['resource_type'], 0)
            if amount > 0:
                yield TransferAction(
                    path=self._transfer_paths[plan['pos']],
                    resource=plan['resource_type'],
                    amount=amount,
                )

        # step 4: claim win if goal reached
        if self.game_state.base.storage.at_least(self.game_state.goal_resources):
            yield ClaimWinAction(x=self._base_pos[0], y=self._base_pos[1])