"""
Smart strategy for Stages 1–3.

Handles all three goal types:
  - STONE: mine stone quarries until target reached (Stage 1)
  - IRON_ORE: mine iron ore and deliver directly to BASE (Stage 2 / Stage 3 variant)
  - IRON: mine iron ore, smelt into iron via smelters, deliver iron to BASE (Stage 3)

Architecture overview:
  1. find_all_paths     — discovers optimal routes to all resource nodes using 0-1 BFS
  2. find_optimal_plan  — runs goal-specific simulation to select minimum structures needed
  3. SmartStrategy      — executes the plan greedily, building each structure as soon
                          as it can actually afford it using real current stone balance

The simulation/execution design principle:
  Simulation uses _actual_ dynamic costs (recalculated as roads are built), which
  exactly mirrors what execution's _actual_cost() computes at build time. This
  eliminates the mismatch bugs that arise from using static pre-estimated costs.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Generator

from game_simulation.actions import (
    BaseAction, BuildAction, ClaimWinAction,
    ExtractAction, ProduceAction, TransferAction,
)
from game_simulation.game_types import ResourceType, StructureType, TerrainType
from strategies.base_strategy import BaseStrategy, StrategyFailed

# ---------------------------------------------------------------------------
# Game constants (from stage capabilities)
# ---------------------------------------------------------------------------

MINE_RATE = 5          # stone per quarry activation / ore per mine activation
SMELTER_RATE = 2       # max IRON_ORE consumed per smelter per PRODUCE action
QUARRY_COST = 10       # STONE to build a STONE_QUARRY
MINE_COST = 15         # STONE to build an IRON_MINE
SMELTER_COST = 20      # STONE to build a SMELTER
SMELTERS_PER_MINE = 3  # max smelters we place per iron mine (ceil(MINE_RATE / SMELTER_RATE))


# ---------------------------------------------------------------------------
# 0-1 BFS path finder
# ---------------------------------------------------------------------------

def _bfs(game_state, targets, network, blocked):
    """
    Find shortest path from BASE to the nearest tile in `targets` using 0-1 BFS.

    Cost model:
      - Traversing a tile already in `network` costs 0 (road already built).
      - Traversing a new buildable tile costs 1 (needs a road built).

    Tiles in `blocked` (extraction structures already placed) cannot be used
    as intermediates — only as final destinations if they are in `targets`.

    Resource node tiles that are NOT in `targets` are also blocked as
    intermediates because roads cannot be built on resource nodes.

    Args:
        game_state:  Current GameState.
        targets:     Set of (x, y) positions we are trying to reach.
        network:     Set of (x, y) positions that already have structures
                     (roads or otherwise) — free to traverse.
        blocked:     Set of (x, y) positions that cannot be path intermediates
                     (e.g. already-placed quarries/mines).

    Returns:
        (path, pos) where path is a list of (x, y) from BASE to the found
        target, or (None, None) if unreachable.
    """
    board = game_state.board
    W, H = board.width, board.height
    grid = board.grid
    base = (game_state.base.x, game_state.base.y)
    rnodes = {(n.x, n.y) for n in board.resource_nodes}
    reachable = targets - {base}
    if not reachable:
        return None, None

    def ok(x, y):
        """Return True if tile (x, y) can be visited by BFS."""
        if not (0 <= x < W and 0 <= y < H):
            return False
        if (x, y) in reachable:
            return True   # always allow destination tiles
        if (x, y) in blocked:
            return False  # extraction structures block intermediate traversal
        if (x, y) in rnodes:
            return False  # resource nodes that aren't our target block roads
        if (x, y) in network:
            return True   # existing network tile: free to traverse
        return grid[y][x] in (TerrainType.GRASS.value, TerrainType.PLANNED_ROAD.value)

    dist = {base: 0}
    parent = {base: None}
    q = deque([(0, base)])
    while q:
        d, (x, y) = q.popleft()
        if d > dist.get((x, y), float('inf')):
            continue  # stale entry — skip
        if (x, y) in reachable:
            # reconstruct path from BASE to this node
            path, cur = [], (x, y)
            while cur:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path, (x, y)
        for nx, ny in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if not ok(nx, ny):
                continue
            c = 0 if (nx, ny) in network else 1  # 0 = free, 1 = new road needed
            nd = d + c
            if nd < dist.get((nx, ny), float('inf')):
                dist[(nx, ny)] = nd
                parent[(nx, ny)] = (x, y)
                # 0-cost edges go to the front (deque left), 1-cost to back
                if c == 0:
                    q.appendleft((nd, (nx, ny)))
                else:
                    q.append((nd, (nx, ny)))
    return None, None


# ---------------------------------------------------------------------------
# Path discovery for all resource nodes and smelter positions
# ---------------------------------------------------------------------------

def find_all_paths(game_state):
    """
    Discover routes from BASE to every resource node and find smelter placements.

    Stone quarry paths:
      Uses a shared, growing network. Each quarry's road tiles are added to the
      network before the next quarry is found, so subsequent quarries benefit
      from already-built roads (lower new-road counts). Discovery order is the
      correct build order — each plan's cost assumes all previous plans' roads
      already exist.

    Iron mine paths:
      Routing reuses the full stone network (cheap paths), but the stored cost
      is calculated from base_network only (conservative). The actual cost at
      execution time is recomputed dynamically by _actual_cost().

    Smelter positions:
      Up to SMELTERS_PER_MINE smelters are placed adjacent to each mine's path,
      as close to BASE as possible (walking from the BASE end toward the mine).
      Candidates are blocked from all planned road tiles, quarry/mine positions,
      resource nodes, and already-used smelter positions to prevent conflicts.

    Returns:
        dict with keys:
          'stone'        — list of stone quarry plan dicts
          'iron'         — list of iron mine plan dicts
          'smelter'      — list of smelter plan dicts
          'base_network' — set of (x, y) occupied by BASE at planning time

    Each plan dict contains:
      'path'          — list of (x, y) from BASE to structure position
      'pos'           — (x, y) where the structure is placed
      'cost'          — estimated stone cost (roads + structure base cost)
      'structure_type'— StructureType enum value
      'resource_type' — ResourceType enum value

    Smelter plans additionally contain:
      'mine_to_smelter_path' — path from mine to smelter (for IRON_ORE transfer)
      'iron_mine_pos'        — (x, y) of the mine this smelter feeds
      'iron_mine_idx'        — index of the mine in iron_plans list
    """
    base_pos = (game_state.base.x, game_state.base.y)
    base_network = set(game_state.structures.keys())
    board = game_state.board
    rnodes = {(n.x, n.y) for n in board.resource_nodes}

    # --- stone quarry paths (shared growing network) ---
    stone_plans = []
    s_net = set(base_network)   # grows as each quarry is added
    s_blocked = set()           # quarry positions (blocked as intermediates)
    snodes = {(n.x,n.y) for n in board.resource_nodes if n.resource == ResourceType.STONE.value}
    while snodes:
        path, pos = _bfs(game_state, snodes, s_net, s_blocked)
        if not path:
            break
        new_roads = [p for p in path[1:-1] if p not in s_net]
        stone_plans.append({
            'path': path, 'pos': pos,
            'cost': len(new_roads) + QUARRY_COST,
            'structure_type': StructureType.STONE_QUARRY,
            'resource_type': ResourceType.STONE,
        })
        # expand network so the next quarry search benefits from these roads
        for p in path[1:-1]: s_net.add(p)
        s_blocked.add(pos)
        s_net.add(pos)
        snodes.discard(pos)

    # --- iron mine paths (routing via full stone network) ---
    iron_plans = []
    i_net = set(s_net)      # inherit all stone roads for routing
    i_blocked = set(s_blocked)
    inodes = {(n.x,n.y) for n in board.resource_nodes if n.resource == ResourceType.IRON_ORE.value}
    while inodes:
        path, pos = _bfs(game_state, inodes, i_net, i_blocked)
        if not path:
            break
        # cost measured from base_network (conservative) — dynamically recomputed at execution
        new_roads = [p for p in path[1:-1] if p not in base_network]
        iron_plans.append({
            'path': path, 'pos': pos,
            'cost': len(new_roads) + MINE_COST,
            'structure_type': StructureType.IRON_MINE,
            'resource_type': ResourceType.IRON_ORE,
        })
        for p in path[1:-1]: i_net.add(p)
        i_blocked.add(pos)
        i_net.add(pos)
        inodes.discard(pos)

    # --- smelter position discovery ---
    # build a master set of all tiles that will be occupied by roads or structures
    # so we never place a smelter on a tile that will have a road through it
    all_planned = set(base_network)
    for p in stone_plans:
        for pos in p['path'][1:-1]: all_planned.add(pos)
        all_planned.add(p['pos'])
    for p in iron_plans:
        for pos in p['path'][1:-1]: all_planned.add(pos)
        all_planned.add(p['pos'])
    all_planned |= rnodes  # resource nodes cannot have smelters either

    smelter_plans = []
    used_smelter = set()  # prevents two smelters at the same position

    for mi, iron_plan in enumerate(iron_plans):
        mine_path = iron_plan['path']
        placed = 0
        # walk from BASE end toward mine — smelters closer to BASE are cheaper
        for i in range(1, len(mine_path) - 1):
            if placed >= SMELTERS_PER_MINE:
                break
            bx, by = mine_path[i]
            for nx, ny in [(bx+1,by),(bx-1,by),(bx,by+1),(bx,by-1)]:
                if placed >= SMELTERS_PER_MINE:
                    break
                c = (nx, ny)
                if c in all_planned or c in used_smelter:
                    continue
                if not (0 <= nx < board.width and 0 <= ny < board.height):
                    continue
                if board.grid[ny][nx] not in (TerrainType.GRASS.value, TerrainType.PLANNED_ROAD.value):
                    continue
                # path BASE → smelter: follow mine path to branch point, then branch off
                path_to_s = mine_path[:i+1] + [c]
                # path mine → smelter: reverse mine path from mine to branch point, then branch
                mine_to_s = list(reversed(mine_path[i:])) + [c]
                new_roads = [p for p in path_to_s[1:-1] if p not in base_network]
                smelter_plans.append({
                    'path': path_to_s,
                    'mine_to_smelter_path': mine_to_s,
                    'pos': c,
                    'cost': len(new_roads) + SMELTER_COST,
                    'structure_type': StructureType.SMELTER,
                    'resource_type': ResourceType.IRON,
                    'iron_mine_pos': iron_plan['pos'],
                    'iron_mine_idx': mi,
                })
                used_smelter.add(c)
                all_planned.add(c)
                placed += 1

    return {
        'stone': stone_plans, 'iron': iron_plans,
        'smelter': smelter_plans, 'base_network': base_network,
    }


def compute_mine_costs(iron_plans, stone_plans, nq, base_network):
    """
    Compute iron mine costs assuming the first `nq` stone quarries have been built.

    When nq quarries are selected, their road tiles are treated as already
    existing (free). This is used by sim_iron to estimate mine affordability
    given a specific quarry subset, allowing the optimiser to find combinations
    where quarry roads reduce mine costs.

    Args:
        iron_plans:   All iron mine plan dicts from find_all_paths.
        stone_plans:  All stone quarry plan dicts from find_all_paths.
        nq:           Number of quarries assumed already built.
        base_network: Set of (x, y) positions BASE occupies at game start.

    Returns:
        List of integer costs, one per iron plan, in the same order.
    """
    avail = set(base_network)
    for p in stone_plans[:nq]:
        for pos in p['path'][1:-1]:
            avail.add(pos)
    return [
        len([p for p in plan['path'][1:-1] if p not in avail]) + MINE_COST
        for plan in iron_plans
    ]


# ---------------------------------------------------------------------------
# Goal-specific simulation functions
# ---------------------------------------------------------------------------

def sim_stone(stone, target, turns, qc):
    """
    Simulate a STONE-goal level.

    Greedily builds quarries (minimum first, cheapest discovery order) and
    mines them until cumulative stone reaches `target`. Called with increasing
    quarry counts until a feasible plan is found.

    Args:
        stone:  Starting stone at BASE.
        target: Stone amount required at BASE to win.
        turns:  Maximum turns allowed.
        qc:     List of quarry costs in build order.

    Returns:
        List of (turn, ResourceType.STONE, quarry_index) build events if
        reachable within turns, else None.
    """
    built = 0
    nq = 0
    events = []
    for turn in range(turns):
        stone += built * MINE_RATE
        if stone >= target:
            return events
        changed = True
        while changed:
            changed = False
            if nq < len(qc) and stone >= qc[nq]:
                stone -= qc[nq]
                events.append((turn, ResourceType.STONE, nq))
                built += 1; nq += 1
                stone += MINE_RATE  # quarry mines immediately on build turn
                changed = True
        if stone >= target:
            return events
    return None


def sim_iron_ore(stone, target, turns, stone_plans, iron_plans):
    """
    Simulate an IRON_ORE-goal level with dynamic mine cost tracking.

    Key design: mine costs are recomputed each time a quarry is built, because
    newly built quarry roads reduce the number of roads the mine still needs.
    This exactly mirrors _actual_cost() in execution, eliminating
    simulation/execution mismatches.

    Both quarries and mines use independent `if` guards (not `elif`) so both
    can be built in the same turn when affordable — critical for high ore
    targets that require many mines built as quickly as possible.

    All mined ore is assumed to go directly to BASE each turn (no smelters
    needed for IRON_ORE goals).

    Args:
        stone:        Starting stone at BASE.
        target:       Iron ore amount required at BASE to win.
        turns:        Maximum turns allowed.
        stone_plans:  Stone quarry plan dicts (in build order).
        iron_plans:   Iron mine plan dicts (in build order).

    Returns:
        List of (turn, ResourceType, index) build events if reachable, else None.
    """
    ore = 0
    bq = bm = nq = nm = 0
    ev = []
    available_roads = set()  # grows as quarries are built — used to recompute mine costs

    qc = [p['cost'] for p in stone_plans]

    def mine_cost(plan):
        """Compute this mine's actual cost given currently available road tiles."""
        new = [p for p in plan['path'][1:-1] if p not in available_roads]
        return len(new) + MINE_COST

    for turn in range(turns):
        stone += bq * MINE_RATE
        ore += bm * MINE_RATE
        if ore >= target:
            return ev
        changed = True
        while changed:
            changed = False
            # try next quarry (independent if — can build quarry AND mine same turn)
            if nq < len(qc) and stone >= qc[nq]:
                stone -= qc[nq]
                ev.append((turn, ResourceType.STONE, nq))
                # update available roads so subsequent mine_cost() calls are accurate
                for pos in stone_plans[nq]['path'][1:-1]:
                    available_roads.add(pos)
                bq += 1; nq += 1
                stone += MINE_RATE
                changed = True
            # try next mine using dynamically computed cost
            if nm < len(iron_plans):
                mc = mine_cost(iron_plans[nm])
                if stone >= mc:
                    stone -= mc
                    ev.append((turn, ResourceType.IRON_ORE, nm))
                    ore += MINE_RATE  # mine produces immediately on build turn
                    bm += 1; nm += 1
                    changed = True
        if ore >= target:
            return ev
    return None


def sim_iron(stone, target, turns, qc, mc, smc, smi):
    """
    Simulate an IRON-goal level (Stage 3).

    Production chain per turn:
      1. Quarries produce stone.
      2. Mines produce ore (stored per-mine to track individual mine capacity).
      3. Each smelter consumes up to SMELTER_RATE ore from its assigned mine,
         subject to a per-mine cap of MINE_RATE total ore consumed per turn.
         This prevents over-counting when multiple smelters share one mine.
      4. Greedily build: quarry → mine (only when first smelter also affordable) → smelter.

    Mine-smelter pairing: smelters are assigned to mines via `smi` (index list).
    A mine is only built when we can also afford its first paired smelter, preventing
    ore from piling up unused while we save for a smelter.

    Args:
        stone:  Starting stone at BASE.
        target: Iron amount required at BASE to win.
        turns:  Maximum turns allowed.
        qc:     Quarry costs in build order.
        mc:     Mine costs in build order (pre-computed for a specific quarry subset).
        smc:    Smelter costs in build order.
        smi:    Smelter-to-mine index mapping (smi[i] = which mine smelter i feeds).

    Returns:
        List of (turn, ResourceType, index) build events if reachable, else None.
    """
    ore = [0] * max(len(mc), 1)  # ore inventory per mine
    iron = 0
    bq = bm = bs = 0
    nq = nm = ns = 0
    events = []

    for turn in range(turns):
        # step 1: quarries produce stone
        stone += bq * MINE_RATE

        # step 2: each iron mine produces ore into its own slot
        for i in range(bm):
            ore[i] += MINE_RATE

        # step 3: smelters consume ore and produce iron
        # mu tracks total ore consumed from each mine this turn (enforces MINE_RATE cap)
        mu = {}
        for i in range(bs):
            mi = smi[i] if i < len(smi) else 0
            mi = mi % max(bm, 1) if bm > 0 else 0
            if mi >= bm:
                continue
            used = mu.get(mi, 0)
            avail = min(ore[mi], SMELTER_RATE, MINE_RATE - used)  # cap at mine rate
            if avail >= 1:
                iron += avail
                ore[mi] -= avail
                mu[mi] = used + avail

        if iron >= target:
            return events

        # step 4: greedy build
        changed = True
        while changed:
            changed = False
            # quarry has highest priority
            if nq < len(qc) and stone >= qc[nq]:
                stone -= qc[nq]
                events.append((turn, ResourceType.STONE, nq))
                bq += 1; nq += 1
                stone += MINE_RATE
                changed = True
            # mine: only when we can also afford its first paired smelter
            # this prevents ore sitting idle for many turns while saving for a smelter
            elif nm < len(mc):
                paired = [
                    smc[i] for i in range(ns, len(smc))
                    if i < len(smi) and smi[i] == nm
                ]
                sc_min = min(paired) if paired else 0
                if stone >= mc[nm] + sc_min:
                    stone -= mc[nm]
                    events.append((turn, ResourceType.IRON_ORE, nm))
                    ore[nm] += MINE_RATE
                    bm += 1; nm += 1
                    changed = True
            # smelter: only once its paired mine is built
            elif ns < len(smc) and bm > 0:
                mi = smi[ns] if ns < len(smi) else 0
                mi = mi % max(bm, 1)
                if mi < bm and stone >= smc[ns]:
                    stone -= smc[ns]
                    events.append((turn, ResourceType.IRON, ns))
                    # immediately process available ore on the build turn
                    used = mu.get(mi, 0)
                    avail = min(ore[mi], SMELTER_RATE, MINE_RATE - used)
                    if avail >= 1:
                        iron += avail
                        ore[mi] -= avail
                        mu[mi] = used + avail
                    bs += 1; ns += 1
                    changed = True

        if iron >= target:
            return events

    return None


# ---------------------------------------------------------------------------
# Plan optimiser
# ---------------------------------------------------------------------------

def find_optimal_plan(paths, stone0, rtype, target, turns):
    """
    Find the minimum set of structures needed to reach `target` within `turns`.

    Each goal type uses a different search strategy:

    STONE (Stage 1):
      Tries nq = 1, 2, 3... quarries (min→max) until sim_stone succeeds.
      Minimum quarries = minimum wasted stone.

    IRON_ORE (Stage 2 / Stage 3 variant):
      Tries nm = 1, 2... mines × nq = 0, 1... quarries (min→max for both).
      sim_iron_ore recomputes mine costs dynamically as quarries are simulated,
      so the simulation is accurate without pre-estimating quarry road savings.

    IRON (Stage 3):
      Tries ns = 1, 2... smelters × nm = 1, 2... mines × nq = max→0 quarries.
      max→min quarry iteration because more quarry roads reduce mine costs —
      we want to find the feasible plan that uses the most efficient road sharing.
      Smelters are filtered to only include those whose paired mine is selected.

    Args:
        paths:   Return value of find_all_paths().
        stone0:  Starting stone at BASE.
        rtype:   Goal ResourceType (STONE, IRON_ORE, or IRON).
        target:  Amount of goal resource needed at BASE.
        turns:   Maximum turns allowed.

    Returns:
        (selected_stone_plans, selected_iron_plans, selected_smelter_plans)
        or (None, None, None) if no feasible plan exists.
    """
    sp = paths['stone']
    ip = paths['iron']
    smp = paths['smelter']
    bn = paths['base_network']

    # STONE goal: minimum quarries first
    if rtype == ResourceType.STONE:
        for nq in range(1, len(sp)+1):
            qc = [p['cost'] for p in sp[:nq]]
            if sim_stone(stone0, target, turns, qc) is not None:
                print(f"Plan: {nq}Q")
                return sp[:nq], [], []
        return None, None, None

    # IRON_ORE goal: dynamic cost simulation, minimum quarries and mines first
    if rtype == ResourceType.IRON_ORE:
        for nm in range(1, len(ip)+1):
            for nq in range(0, len(sp)+1):
                if sim_iron_ore(stone0, target, turns, sp[:nq], ip[:nm]) is not None:
                    print(f"Plan: {nq}Q+{nm}M")
                    return sp[:nq], ip[:nm], []
        return None, None, None

    # IRON goal: max→min quarries (more roads = cheaper mines)
    for ns in range(1, len(smp)+1):
        for nm in range(1, len(ip)+1):
            for nq in range(len(sp), -1, -1):
                qc = [p['cost'] for p in sp[:nq]]
                mc = compute_mine_costs(ip, sp, nq, bn)[:nm]
                # only include smelters whose paired mine is within the selected set
                vs = [p for p in smp if p['iron_mine_idx'] < nm]
                if len(vs) < ns:
                    continue
                sel = vs[:ns]
                smc = [p['cost'] for p in sel]
                smi = [p['iron_mine_idx'] for p in sel]
                if sim_iron(stone0, target, turns, qc, mc, smc, smi) is not None:
                    print(f"Plan: {nq}Q+{nm}M+{ns}S")
                    return sp[:nq], ip[:nm], sel

    return None, None, None


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class SmartStrategy(BaseStrategy):
    """
    Greedy execution strategy for Stages 1–3.

    __init__ calls find_all_paths and find_optimal_plan to determine WHAT
    and HOW MANY structures to build. No rigid schedule is computed — instead,
    generate_more_turn_actions builds each structure as soon as the actual
    current stone balance can afford it (_actual_cost checks real remaining roads).

    This greedy execution approach is robust to cost estimation errors because
    it never commits to building something it cannot currently afford.

    Turn order each turn:
      1. Mine all quarries → transfer STONE to BASE
      2. Mine all iron mines → transfer IRON_ORE to smelters (or to BASE for IRON_ORE goals)
      3. PRODUCE at all smelters → transfer IRON to BASE
      4. Build next affordable quarry / mine / smelter (in that priority order)
      5. CLAIM_WIN if goal resource target met at BASE
    """

    def __init__(self, game_state):
        super().__init__(game_state)
        self._base_pos = (game_state.base.x, game_state.base.y)
        self._rnodes = {(n.x,n.y) for n in game_state.board.resource_nodes}

        # determine goal type and amount from level definition
        goal = game_state.goal_resources
        if goal.get(ResourceType.IRON, 0) > 0:
            self._rtype = ResourceType.IRON
            target = goal.get(ResourceType.IRON, 0)
        elif goal.get(ResourceType.IRON_ORE, 0) > 0:
            self._rtype = ResourceType.IRON_ORE
            target = goal.get(ResourceType.IRON_ORE, 0)
        else:
            self._rtype = ResourceType.STONE
            target = goal.get(ResourceType.STONE, 0)

        stone0 = game_state.base.storage.get(ResourceType.STONE, 0)
        turns = game_state.max_turns

        paths = find_all_paths(game_state)

        print(f"Stone plans: {len(paths['stone'])}")
        for p in paths['stone']:
            print(f"  Quarry at {p['pos']}, cost={p['cost']}")
        print(f"Iron plans: {len(paths['iron'])}")
        for p in paths['iron']:
            print(f"  Mine at {p['pos']}, cost={p['cost']}")
        print(f"Smelter plans: {len(paths['smelter'])}")
        for p in paths['smelter']:
            print(f"  Smelter at {p['pos']}, cost={p['cost']}, mine={p['iron_mine_pos']}")
        print(f"Stone={stone0}, Goal={target} {self._rtype.value}, Turns={turns}")

        ss, si, ssm = find_optimal_plan(paths, stone0, self._rtype, target, turns)
        if ss is None:
            raise StrategyFailed("No valid plan found.")

        # selected structure plans in build order
        self._ss = ss   # stone quarry plans
        self._si = si   # iron mine plans
        self._ssm = ssm # smelter plans

        # greedy build pointers — advance as each structure is built
        self._nq = self._nm = self._ns = 0

        # built structure tracking (positions)
        self._bq = []  # built quarry positions
        self._bm = []  # built mine positions
        self._bs = []  # built smelter positions

        # transfer path registries (populated as structures are built)
        self._q2b = {}         # quarry_pos → path to BASE (for STONE transfer)
        self._m2s = {}         # (mine_pos, smelter_pos) → path (for IRON_ORE to smelter)
        self._s2b = {}         # smelter_pos → path to BASE (for IRON transfer)
        self._mfs = {}         # smelter_pos → mine_pos it is fed by
        self._mine_to_base = {}# mine_pos → path to BASE (for IRON_ORE goal direct delivery)

        self._last = -1  # last turn we acted — prevents acting twice in one turn

    def _actual_cost(self, plan, base_cost):
        """
        Compute the real stone cost to build this structure right now.

        Counts only road tiles in the plan's path that do not yet have a
        structure on them (i.e. still need a road built). Resource node tiles
        are excluded because roads are never built on resource nodes.

        Args:
            plan:       A plan dict from find_all_paths.
            base_cost:  Structure base cost (QUARRY_COST, MINE_COST, or SMELTER_COST).

        Returns:
            Integer stone cost.
        """
        roads = sum(
            1 for p in plan['path'][1:-1]
            if self.game_state.get_structure_at(*p) is None
            and p not in self._rnodes
        )
        return roads + base_cost

    def generate_more_turn_actions(self) -> Generator[BaseAction, None, None]:
        """
        Yield all actions for the current turn.

        Called repeatedly by the framework within a single turn until it yields
        nothing. The _last guard ensures we yield all actions on the first call
        and nothing on subsequent calls within the same turn.
        """
        t = self.game_state.executed_turns
        if self._last == t:
            return  # already acted this turn — yield nothing to end the turn
        self._last = t

        # --- step 1: mine all quarries → transfer STONE to BASE ---
        for q in self._bq:
            yield ExtractAction(x=q[0], y=q[1])
        for q in self._bq:
            s = self.game_state.get_structure_at(*q)
            amt = s.storage.get(ResourceType.STONE, 0)
            if amt > 0:
                yield TransferAction(path=self._q2b[q],
                                     resource=ResourceType.STONE, amount=amt)

        # --- step 2: mine all iron mines ---
        for m in self._bm:
            yield ExtractAction(x=m[0], y=m[1])

        # transfer ore: directly to BASE for IRON_ORE goals, to smelters for IRON goals
        for m in self._bm:
            mine = self.game_state.get_structure_at(*m)
            ore = mine.storage.get(ResourceType.IRON_ORE, 0)
            if ore <= 0:
                continue
            if self._rtype == ResourceType.IRON_ORE:
                # IRON_ORE goal: mine delivers directly to BASE
                path = self._mine_to_base.get(m)
                if path and len(path) >= 3:
                    yield TransferAction(path=path,
                                         resource=ResourceType.IRON_ORE, amount=ore)
            else:
                # IRON goal: distribute ore to assigned smelters (capped at SMELTER_RATE each)
                for sm in [s for s in self._bs if self._mfs.get(s) == m]:
                    if ore <= 0:
                        break
                    t2 = min(ore, SMELTER_RATE)
                    path = self._m2s.get((m, sm))
                    if path and len(path) >= 3:
                        yield TransferAction(path=path,
                                             resource=ResourceType.IRON_ORE, amount=t2)
                        ore -= t2

        # --- step 3: produce at smelters → transfer IRON to BASE ---
        for sm in self._bs:
            s = self.game_state.get_structure_at(*sm)
            if s.storage.get(ResourceType.IRON_ORE, 0) > 0 and s.can_produce:
                yield ProduceAction(x=sm[0], y=sm[1])
            fe = s.storage.get(ResourceType.IRON, 0)
            if fe > 0:
                yield TransferAction(path=self._s2b[sm],
                                     resource=ResourceType.IRON, amount=fe)

        # --- step 4: greedy build ---
        # Build priority: quarry > mine > smelter
        # Each structure is built as soon as actual current cost is affordable.
        # 'continue' restarts the loop so that after building one structure we
        # immediately check if we can now afford the next.
        changed = True
        while changed:
            changed = False
            stone = self.game_state.base.storage.get(ResourceType.STONE, 0)

            # quarry
            if self._nq < len(self._ss):
                plan = self._ss[self._nq]
                if stone >= self._actual_cost(plan, QUARRY_COST):
                    yield from self._do_quarry(plan)
                    self._nq += 1
                    changed = True
                    continue

            # mine
            if self._nm < len(self._si):
                plan = self._si[self._nm]
                mc = self._actual_cost(plan, MINE_COST)
                if self._rtype == ResourceType.IRON_ORE:
                    # IRON_ORE: build as soon as affordable (no smelter needed)
                    if stone >= mc:
                        yield from self._do_mine(plan)
                        self._nm += 1
                        changed = True
                        continue
                else:
                    # IRON: require mine + first smelter to be jointly affordable
                    # so ore never piles up waiting for a smelter
                    paired = [
                        p for p in self._ssm
                        if p['iron_mine_pos'] == plan['pos']
                        and p['pos'] not in self._bs
                    ]
                    sc = self._actual_cost(paired[0], SMELTER_COST) if paired else 0
                    if stone >= mc + sc:
                        yield from self._do_mine(plan)
                        self._nm += 1
                        changed = True
                        continue

            # smelter (only once its paired mine exists)
            if self._ns < len(self._ssm):
                plan = self._ssm[self._ns]
                if plan['iron_mine_pos'] in self._bm:
                    if stone >= self._actual_cost(plan, SMELTER_COST):
                        yield from self._do_smelter(plan)
                        self._ns += 1
                        changed = True
                        continue

        # --- step 5: claim win if goal is met ---
        if self.game_state.base.storage.at_least(self.game_state.goal_resources):
            yield ClaimWinAction(x=self._base_pos[0], y=self._base_pos[1])

    def _build_roads(self, plan):
        """
        Yield BuildAction(ROAD) for each tile in plan's path that still needs one.

        Skips tiles that already have a structure and skips resource node tiles
        (which cannot have roads built on them).
        """
        for pos in plan['path'][1:-1]:
            if (self.game_state.get_structure_at(*pos) is None
                    and pos not in self._rnodes):
                yield BuildAction(x=pos[0], y=pos[1],
                                  structure_type=StructureType.ROAD)

    def _do_quarry(self, plan):
        """
        Build roads + STONE_QUARRY, then immediately mine and transfer to BASE.

        Mining immediately on the build turn matches sim_stone's assumption that
        the quarry mines on the same turn it is built.
        """
        yield from self._build_roads(plan)
        yield BuildAction(x=plan['pos'][0], y=plan['pos'][1],
                          structure_type=StructureType.STONE_QUARRY)
        self._q2b[plan['pos']] = list(reversed(plan['path']))
        self._bq.append(plan['pos'])
        yield ExtractAction(x=plan['pos'][0], y=plan['pos'][1])
        q = self.game_state.get_structure_at(*plan['pos'])
        amt = q.storage.get(ResourceType.STONE, 0)
        if amt > 0:
            yield TransferAction(path=self._q2b[plan['pos']],
                                 resource=ResourceType.STONE, amount=amt)

    def _do_mine(self, plan):
        """
        Build roads + IRON_MINE, then immediately mine.

        For IRON_ORE goals: stores mine→BASE path and transfers ore directly.
        For IRON goals: if a smelter already exists for this mine, transfers ore
        to it and triggers production immediately.
        """
        yield from self._build_roads(plan)
        yield BuildAction(x=plan['pos'][0], y=plan['pos'][1],
                          structure_type=StructureType.IRON_MINE)
        self._bm.append(plan['pos'])

        # store mine→BASE path for IRON_ORE goals
        if self._rtype == ResourceType.IRON_ORE:
            self._mine_to_base[plan['pos']] = list(reversed(plan['path']))

        yield ExtractAction(x=plan['pos'][0], y=plan['pos'][1])
        mine = self.game_state.get_structure_at(*plan['pos'])
        ore = mine.storage.get(ResourceType.IRON_ORE, 0)

        if self._rtype == ResourceType.IRON_ORE:
            path = self._mine_to_base.get(plan['pos'])
            if ore > 0 and path and len(path) >= 3:
                yield TransferAction(path=path,
                                     resource=ResourceType.IRON_ORE, amount=ore)
        else:
            # if smelter(s) already built for this mine, feed them immediately
            for sm in [s for s in self._bs if self._mfs.get(s) == plan['pos']]:
                if ore <= 0:
                    break
                t = min(ore, SMELTER_RATE)
                path = self._m2s.get((plan['pos'], sm))
                if path and len(path) >= 3:
                    yield TransferAction(path=path,
                                         resource=ResourceType.IRON_ORE, amount=t)
                    ore -= t
                    s = self.game_state.get_structure_at(*sm)
                    if s.storage.get(ResourceType.IRON_ORE, 0) > 0 and s.can_produce:
                        yield ProduceAction(x=sm[0], y=sm[1])
                        fe = s.storage.get(ResourceType.IRON, 0)
                        if fe > 0:
                            yield TransferAction(path=self._s2b[sm],
                                                 resource=ResourceType.IRON, amount=fe)

    def _do_smelter(self, plan):
        """
        Build roads + SMELTER, register it, and immediately process any
        available ore from its paired mine.

        Stores:
          _s2b[smelter_pos]            — path to BASE for IRON delivery
          _mfs[smelter_pos]            — which mine feeds this smelter
          _m2s[(mine_pos, smelter_pos)]— path for IRON_ORE delivery to smelter
        """
        yield from self._build_roads(plan)
        yield BuildAction(x=plan['pos'][0], y=plan['pos'][1],
                          structure_type=StructureType.SMELTER)
        sp2, mp = plan['pos'], plan['iron_mine_pos']
        self._bs.append(sp2)
        self._s2b[sp2] = list(reversed(plan['path']))
        self._mfs[sp2] = mp
        self._m2s[(mp, sp2)] = plan['mine_to_smelter_path']

        # if mine already built, immediately transfer available ore and produce
        if mp in self._bm:
            mine = self.game_state.get_structure_at(*mp)
            ore = mine.storage.get(ResourceType.IRON_ORE, 0)
            t = min(ore, SMELTER_RATE)
            path = plan['mine_to_smelter_path']
            if t > 0 and len(path) >= 3:
                yield TransferAction(path=path,
                                     resource=ResourceType.IRON_ORE, amount=t)
                s = self.game_state.get_structure_at(*sp2)
                if s.storage.get(ResourceType.IRON_ORE, 0) > 0 and s.can_produce:
                    yield ProduceAction(x=sp2[0], y=sp2[1])
                    fe = s.storage.get(ResourceType.IRON, 0)
                    if fe > 0:
                        yield TransferAction(path=self._s2b[sp2],
                                             resource=ResourceType.IRON, amount=fe)