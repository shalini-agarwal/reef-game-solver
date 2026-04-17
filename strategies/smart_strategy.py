"""
Smart strategy for Stages 1-3.
Core principle: simulation finds WHAT and HOW MANY structures to build,
execution builds them greedily using actual current costs.
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

MINE_RATE = 5
SMELTER_RATE = 2
QUARRY_COST = 10
MINE_COST = 15
SMELTER_COST = 20
SMELTERS_PER_MINE = 3


def _bfs(game_state, targets, network, blocked):
    """
    0-1 BFS from BASE to nearest target.
    network tiles cost 0, new tiles cost 1.
    blocked tiles (extraction structures, resource nodes) cannot be intermediates.
    Returns (path, pos) or (None, None).
    """
    board = game_state.board
    W, H = board.width, board.height
    grid = board.grid
    base = (game_state.base.x, game_state.base.y)
    resource_nodes = {(n.x, n.y) for n in board.resource_nodes}

    reachable = targets - {base}
    if not reachable:
        return None, None

    def ok(x, y):
        if not (0 <= x < W and 0 <= y < H):
            return False
        if (x, y) in reachable:
            return True
        if (x, y) in blocked:
            return False
        if (x, y) in resource_nodes:
            return False
        if (x, y) in network:
            return True
        return grid[y][x] in (TerrainType.GRASS.value, TerrainType.PLANNED_ROAD.value)

    dist = {base: 0}
    parent = {base: None}
    q = deque([(0, base)])

    while q:
        d, (x, y) = q.popleft()
        if d > dist.get((x, y), float('inf')):
            continue
        if (x, y) in reachable:
            path, cur = [], (x, y)
            while cur:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path, (x, y)
        for nx, ny in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if not ok(nx, ny):
                continue
            c = 0 if (nx, ny) in network else 1
            nd = d + c
            if nd < dist.get((nx, ny), float('inf')):
                dist[(nx, ny)] = nd
                parent[(nx, ny)] = (x, y)
                if c == 0:
                    q.appendleft((nd, (nx, ny)))
                else:
                    q.append((nd, (nx, ny)))
    return None, None


def find_all_paths(game_state):
    """
    Find paths to all resource nodes and smelter positions.
    
    Stone quarries: shared growing network (discovery order = optimal build order).
    Iron mines: routing via stone network, costs stored from base_network
                (recalculated per quarry subset in find_optimal_plan).
    Smelters: placed adjacent to mine paths, blocked from all planned tiles.
    """
    base_pos = (game_state.base.x, game_state.base.y)
    base_network = set(game_state.structures.keys())
    board = game_state.board
    resource_nodes = {(n.x, n.y) for n in board.resource_nodes}

    # --- stone quarry paths (shared network) ---
    stone_plans = []
    s_net = set(base_network)
    s_blocked = set()
    s_done = set()
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
        s_done.add(pos)
        for p in path[1:-1]: s_net.add(p)
        s_blocked.add(pos)
        s_net.add(pos)
        snodes.discard(pos)

    # --- iron mine paths (routing via stone network, costs from base_network) ---
    iron_plans = []
    i_net = set(s_net)
    i_blocked = set(s_blocked)
    i_done = set()
    inodes = {(n.x,n.y) for n in board.resource_nodes if n.resource == ResourceType.IRON_ORE.value}

    while inodes:
        path, pos = _bfs(game_state, inodes, i_net, i_blocked)
        if not path:
            break
        # cost uses base_network — recalculated per quarry subset in simulation
        new_roads = [p for p in path[1:-1] if p not in base_network]
        iron_plans.append({
            'path': path, 'pos': pos,
            'cost': len(new_roads) + MINE_COST,
            'structure_type': StructureType.IRON_MINE,
            'resource_type': ResourceType.IRON_ORE,
        })
        i_done.add(pos)
        for p in path[1:-1]: i_net.add(p)
        i_blocked.add(pos)
        i_net.add(pos)
        inodes.discard(pos)

    # --- smelter positions ---
    # block ALL planned road tiles and structure positions
    all_planned = set(base_network)
    for p in stone_plans:
        for pos in p['path'][1:-1]: all_planned.add(pos)
        all_planned.add(p['pos'])
    for p in iron_plans:
        for pos in p['path'][1:-1]: all_planned.add(pos)
        all_planned.add(p['pos'])
    all_planned |= resource_nodes

    smelter_plans = []
    used_smelter = set()

    for mi, iron_plan in enumerate(iron_plans):
        mine_path = iron_plan['path']
        path_set = set(map(tuple, mine_path))
        placed = 0

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
                path_to_s = mine_path[:i+1] + [c]
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
    Mine costs assuming the first nq quarries are built.
    Their road tiles are treated as already existing (free).
    This matches execution reality: quarries are built before mines.
    """
    avail = set(base_network)
    for p in stone_plans[:nq]:
        for pos in p['path'][1:-1]:
            avail.add(pos)
    return [
        len([p for p in plan['path'][1:-1] if p not in avail]) + MINE_COST
        for plan in iron_plans
    ]

def simulate(stone, target, rtype, turns, sc, mc, smc, smi):
    ore = [0] * max(len(mc), 1)
    iron = 0
    ore_at_base = 0   # ← add this: tracks IRON_ORE delivered to BASE
    bq = bm = bs = 0
    nq = nm = ns = 0
    events = []

    stone_goal = target if rtype == ResourceType.STONE else 0
    ore_goal = target if rtype == ResourceType.IRON_ORE else 0
    iron_goal = target if rtype == ResourceType.IRON else 0

    for turn in range(turns):
        # mine quarries
        stone += bq * MINE_RATE

        # mine iron mines
        for i in range(bm):
            ore[i] += MINE_RATE

        # for IRON_ORE goal: transfer all ore directly to base each turn
        if rtype == ResourceType.IRON_ORE:
            for i in range(bm):
                ore_at_base += ore[i]
                ore[i] = 0

        # smelt (only for IRON goal)
        mu = {}
        if rtype == ResourceType.IRON:
            for i in range(bs):
                mi = smi[i] % max(bm, 1) if bm > 0 else 0
                if mi >= bm:
                    continue
                used = mu.get(mi, 0)
                avail = min(ore[mi], SMELTER_RATE, MINE_RATE - used)
                if avail >= 1:
                    iron += avail
                    ore[mi] -= avail
                    mu[mi] = used + avail

        # check win
        if (stone >= stone_goal
                and ore_at_base >= ore_goal
                and iron >= iron_goal):
            return events

        # build greedily
        changed = True
        while changed:
            changed = False
            s = stone

            if nq < len(sc) and s >= sc[nq]:
                stone -= sc[nq]
                events.append((turn, ResourceType.STONE, nq))
                bq += 1; nq += 1
                stone += MINE_RATE
                changed = True

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
                    # immediately transfer to base for IRON_ORE goal
                    if rtype == ResourceType.IRON_ORE:
                        ore_at_base += ore[nm]
                        ore[nm] = 0
                    bm += 1; nm += 1
                    changed = True

            elif ns < len(smc) and bm > 0:
                mi = smi[ns] % max(bm, 1) if ns < len(smi) else 0
                if mi < bm and stone >= smc[ns]:
                    stone -= smc[ns]
                    events.append((turn, ResourceType.IRON, ns))
                    used = mu.get(mi, 0)
                    avail = min(ore[mi], SMELTER_RATE, MINE_RATE - used)
                    if avail >= 1:
                        iron += avail; ore[mi] -= avail
                        mu[mi] = used + avail
                    bs += 1; ns += 1
                    changed = True

    return None


def find_optimal_plan(paths, stone0, rtype, target, turns):
    """
    Find minimum structures to reach target within turns.
    Iterates nq from MAX→0 so mines benefit from most quarry roads.
    """
    sp = paths['stone']
    ip = paths['iron']
    smp = paths['smelter']
    bn = paths['base_network']

    if rtype == ResourceType.STONE:
        for nq in range(1, len(sp)+1):
            sc = [p['cost'] for p in sp[:nq]]
            if simulate(stone0, target, rtype, turns, sc, [], [], []) is not None:
                print(f"Plan: {nq}Q")
                return sp[:nq], [], []
        return None, None, None

    if rtype == ResourceType.IRON_ORE:
        for nm in range(1, len(ip)+1):
            for nq in range(len(sp), -1, -1):
                sc = [p['cost'] for p in sp[:nq]]
                mc = compute_mine_costs(ip, sp, nq, bn)[:nm]
                if simulate(stone0, target, rtype, turns, sc, mc, [], []) is not None:
                    print(f"Plan: {nq}Q+{nm}M")
                    return sp[:nq], ip[:nm], []
        return None, None, None

    # IRON goal
    for ns in range(1, len(smp)+1):
        for nm in range(1, len(ip)+1):
            for nq in range(len(sp), -1, -1):
                sc = [p['cost'] for p in sp[:nq]]
                mc = compute_mine_costs(ip, sp, nq, bn)[:nm]
                vs = [p for p in smp if p['iron_mine_idx'] < nm]
                if len(vs) < ns:
                    continue
                sel = vs[:ns]
                smc = [p['cost'] for p in sel]
                smi = [p['iron_mine_idx'] for p in sel]
                if simulate(stone0, target, rtype, turns, sc, mc, smc, smi) is not None:
                    print(f"Plan: {nq}Q+{nm}M+{ns}S")
                    return sp[:nq], ip[:nm], sel
    return None, None, None


class SmartStrategy(BaseStrategy):

    def __init__(self, game_state):
        super().__init__(game_state)
        self._base_pos = (game_state.base.x, game_state.base.y)
        self._rnodes = {(n.x,n.y) for n in game_state.board.resource_nodes}

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

        self._ss = ss    # selected stone plans
        self._si = si    # selected iron plans
        self._ssm = ssm  # selected smelter plans

        # greedy execution pointers
        self._nq = self._nm = self._ns = 0

        # built tracking
        self._bq = []   # built quarry positions
        self._bm = []   # built mine positions
        self._bs = []   # built smelter positions

        # paths
        self._q2b = {}   # quarry→BASE path
        self._m2s = {}   # (mine,smelter)→path
        self._s2b = {}   # smelter→BASE path
        self._mfs = {}   # smelter→mine mapping

        self._last = -1

    def _actual_cost(self, plan, base_cost):
        """Actual stone cost right now (roads not yet built + structure)."""
        roads = sum(
            1 for p in plan['path'][1:-1]
            if self.game_state.get_structure_at(*p) is None
            and p not in self._rnodes
        )
        return roads + base_cost

    def generate_more_turn_actions(self) -> Generator[BaseAction, None, None]:
        t = self.game_state.executed_turns
        if self._last == t:
            return
        self._last = t

        # 1. mine quarries → transfer stone to BASE
        for q in self._bq:
            yield ExtractAction(x=q[0], y=q[1])
        for q in self._bq:
            s = self.game_state.get_structure_at(*q)
            amt = s.storage.get(ResourceType.STONE, 0)
            if amt > 0:
                yield TransferAction(path=self._q2b[q],
                                     resource=ResourceType.STONE, amount=amt)

        # 2. mine iron mines → transfer ore to smelters OR directly to BASE
        for m in self._bm:
            yield ExtractAction(x=m[0], y=m[1])
        for m in self._bm:
            mine = self.game_state.get_structure_at(*m)
            ore = mine.storage.get(ResourceType.IRON_ORE, 0)
            if ore <= 0:
                continue

            if self._rtype == ResourceType.IRON_ORE:
                # transfer directly to BASE
                plan = next(p for p in self._si if p['pos'] == m)
                path_to_base = list(reversed(plan['path']))
                if len(path_to_base) >= 3:
                    yield TransferAction(
                        path=path_to_base,
                        resource=ResourceType.IRON_ORE,
                        amount=ore,
                    )
            else:
                # transfer to assigned smelters
                for sm in [s for s in self._bs if self._mfs.get(s) == m]:
                    if ore <= 0:
                        break
                    t = min(ore, SMELTER_RATE)
                    path = self._m2s.get((m, sm))
                    if path and len(path) >= 3:
                        yield TransferAction(path=path,
                                            resource=ResourceType.IRON_ORE, amount=t)
                        ore -= t

        # 3. produce at smelters → transfer iron to BASE
        for sm in self._bs:
            s = self.game_state.get_structure_at(*sm)
            if s.storage.get(ResourceType.IRON_ORE, 0) > 0 and s.can_produce:
                yield ProduceAction(x=sm[0], y=sm[1])
            fe = s.storage.get(ResourceType.IRON, 0)
            if fe > 0:
                yield TransferAction(path=self._s2b[sm],
                                     resource=ResourceType.IRON, amount=fe)

        # 4. greedy build using actual current costs
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

            # mine — only with enough for mine + its first smelter
            if self._nm < len(self._si):
                plan = self._si[self._nm]
                mc = self._actual_cost(plan, MINE_COST)
                # find cheapest unbuilt smelter for this mine
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

            # smelter
            if self._ns < len(self._ssm):
                plan = self._ssm[self._ns]
                if plan['iron_mine_pos'] in self._bm:
                    if stone >= self._actual_cost(plan, SMELTER_COST):
                        yield from self._do_smelter(plan)
                        self._ns += 1
                        changed = True
                        continue

        # 5. claim win
        if self.game_state.base.storage.at_least(self.game_state.goal_resources):
            yield ClaimWinAction(x=self._base_pos[0], y=self._base_pos[1])

    def _build_roads(self, plan):
        for pos in plan['path'][1:-1]:
            if (self.game_state.get_structure_at(*pos) is None
                    and pos not in self._rnodes):
                yield BuildAction(x=pos[0], y=pos[1],
                                  structure_type=StructureType.ROAD)

    def _do_quarry(self, plan):
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
        yield from self._build_roads(plan)
        yield BuildAction(x=plan['pos'][0], y=plan['pos'][1],
                          structure_type=StructureType.IRON_MINE)
        self._bm.append(plan['pos'])
        yield ExtractAction(x=plan['pos'][0], y=plan['pos'][1])
        # if smelter already built for this mine, transfer and produce
        mine = self.game_state.get_structure_at(*plan['pos'])
        ore = mine.storage.get(ResourceType.IRON_ORE, 0)
        for sm in [s for s in self._bs if self._mfs.get(s) == plan['pos']]:
            if ore <= 0: break
            t = min(ore, SMELTER_RATE)
            path = self._m2s.get((plan['pos'], sm))
            if path and len(path) >= 3:
                yield TransferAction(path=path, resource=ResourceType.IRON_ORE, amount=t)
                ore -= t
                s = self.game_state.get_structure_at(*sm)
                if s.storage.get(ResourceType.IRON_ORE, 0) > 0 and s.can_produce:
                    yield ProduceAction(x=sm[0], y=sm[1])
                    fe = s.storage.get(ResourceType.IRON, 0)
                    if fe > 0:
                        yield TransferAction(path=self._s2b[sm],
                                             resource=ResourceType.IRON, amount=fe)

    def _do_smelter(self, plan):
        yield from self._build_roads(plan)
        yield BuildAction(x=plan['pos'][0], y=plan['pos'][1],
                          structure_type=StructureType.SMELTER)
        sp, mp = plan['pos'], plan['iron_mine_pos']
        self._bs.append(sp)
        self._s2b[sp] = list(reversed(plan['path']))
        self._mfs[sp] = mp
        self._m2s[(mp, sp)] = plan['mine_to_smelter_path']
        # if mine built, transfer and produce immediately
        if mp in self._bm:
            mine = self.game_state.get_structure_at(*mp)
            ore = mine.storage.get(ResourceType.IRON_ORE, 0)
            t = min(ore, SMELTER_RATE)
            path = plan['mine_to_smelter_path']
            if t > 0 and len(path) >= 3:
                yield TransferAction(path=path, resource=ResourceType.IRON_ORE, amount=t)
                s = self.game_state.get_structure_at(*sp)
                if s.storage.get(ResourceType.IRON_ORE, 0) > 0 and s.can_produce:
                    yield ProduceAction(x=sp[0], y=sp[1])
                    fe = s.storage.get(ResourceType.IRON, 0)
                    if fe > 0:
                        yield TransferAction(path=self._s2b[sp],
                                             resource=ResourceType.IRON, amount=fe)