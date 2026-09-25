"""Deterministic generator for order-picking task variants.

Three families of warehouse layouts, each parameterised by a seed:

  open_floor    small empty floor, 3 items          (easy)
  aisle_racks   classic rack/aisle warehouse, 5 items
  bottleneck    two halves joined by one door, 5 items (trips up greedy agents)

A variant is fully determined by its id "<family>:<seed>", so the verifier can
rebuild it from the id alone and never has to trust anything the agent reports.
"""

from __future__ import annotations

import functools
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

Cell = Tuple[int, int]

# action ids
NORTH, SOUTH, WEST, EAST, PICK = 0, 1, 2, 3, 4
MOVES: Dict[int, Cell] = {NORTH: (-1, 0), SOUTH: (1, 0), WEST: (0, -1), EAST: (0, 1)}
ACTION_NAMES = {NORTH: "N", SOUTH: "S", WEST: "W", EAST: "E", PICK: "P"}
N_ACTIONS = 5

FAMILIES = ("open_floor", "aisle_racks", "bottleneck")
SEEDS = (0, 1, 2, 3, 4)

# Safety margin used by the reward design (see README). Generation asserts that
# no variant exceeds it.
MAX_ALLOWED_DISTANCE = 40


@dataclass(frozen=True)
class Variant:
    """An immutable task instance."""

    vid: str
    family: str
    seed: int
    grid: Tuple[Tuple[int, ...], ...]  # 0 = floor, 1 = rack/wall
    depot: Cell
    items: Tuple[Cell, ...]
    budget: int             # max number of actions allowed
    optimal_actions: int    # provably minimal number of actions

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0])

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width

    def is_floor(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and self.grid[cell[0]][cell[1]] == 0

    def render(self, pos: Cell | None = None, picked: Sequence[bool] | None = None) -> str:
        picked = picked if picked is not None else [False] * self.n_items
        rows = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                ch = "#" if self.grid[r][c] else "."
                if (r, c) == self.depot:
                    ch = "D"
                if (r, c) in self.items:
                    ch = "o" if picked[self.items.index((r, c))] else "x"
                if pos is not None and (r, c) == pos:
                    ch = "@"
                row.append(ch)
            rows.append("".join(row))
        return "\n".join(rows)


# --------------------------------------------------------------------------
# grid utilities
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def bfs_distances(grid: Tuple[Tuple[int, ...], ...], src: Cell) -> Dict[Cell, int]:
    """Shortest walking distance (in moves) from src to every reachable floor cell.

    Cached: the returned dict is shared, so callers must treat it as read-only.
    """
    h, w = len(grid), len(grid[0])
    dist = {src: 0}
    q = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in MOVES.values():
            nxt = (r + dr, c + dc)
            if 0 <= nxt[0] < h and 0 <= nxt[1] < w and grid[nxt[0]][nxt[1]] == 0 and nxt not in dist:
                dist[nxt] = dist[(r, c)] + 1
                q.append(nxt)
    return dist


def bfs_move_path(grid: Tuple[Tuple[int, ...], ...], src: Cell, dst: Cell) -> List[int]:
    """A shortest sequence of move actions from src to dst."""
    if src == dst:
        return []
    h, w = len(grid), len(grid[0])
    parent: Dict[Cell, Tuple[Cell, int]] = {src: (src, -1)}
    q = deque([src])
    while q:
        cur = q.popleft()
        if cur == dst:
            break
        for a, (dr, dc) in MOVES.items():
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < h and 0 <= nxt[1] < w and grid[nxt[0]][nxt[1]] == 0 and nxt not in parent:
                parent[nxt] = (cur, a)
                q.append(nxt)
    if dst not in parent:
        raise ValueError(f"no path from {src} to {dst}")
    actions: List[int] = []
    cur = dst
    while cur != src:
        prev, a = parent[cur]
        actions.append(a)
        cur = prev
    actions.reverse()
    return actions


def pairwise_distances(grid, nodes: Sequence[Cell]) -> List[List[int]]:
    table = []
    for node in nodes:
        d = bfs_distances(grid, node)
        table.append([d[other] for other in nodes])
    return table


def held_karp(dist: List[List[int]]) -> Tuple[int, List[int]]:
    """Exact shortest closed tour from node 0 visiting nodes 1..m and returning.

    Returns (cost, visiting order as item indices 0..m-1).
    """
    m = len(dist) - 1
    if m == 0:
        return 0, []
    INF = math.inf
    size = 1 << m
    dp = [[INF] * m for _ in range(size)]
    par = [[-1] * m for _ in range(size)]
    for j in range(m):
        dp[1 << j][j] = dist[0][j + 1]
    for mask in range(size):
        for j in range(m):
            if not (mask >> j) & 1 or dp[mask][j] == INF:
                continue
            base = dp[mask][j]
            for k in range(m):
                if (mask >> k) & 1:
                    continue
                nmask = mask | (1 << k)
                cand = base + dist[j + 1][k + 1]
                if cand < dp[nmask][k]:
                    dp[nmask][k] = cand
                    par[nmask][k] = j
    full = size - 1
    best, best_j = INF, -1
    for j in range(m):
        cand = dp[full][j] + dist[j + 1][0]
        if cand < best:
            best, best_j = cand, j
    order: List[int] = []
    mask, j = full, best_j
    while j != -1:
        order.append(j)
        prev = par[mask][j]
        mask ^= 1 << j
        j = prev
    order.reverse()
    return int(best), order


# --------------------------------------------------------------------------
# layouts
# --------------------------------------------------------------------------

def _layout_open_floor() -> Tuple[List[List[int]], Cell, int]:
    h, w = 7, 7
    grid = [[0] * w for _ in range(h)]
    return grid, (0, 0), 3


def _layout_aisle_racks() -> Tuple[List[List[int]], Cell, int]:
    h, w = 9, 11
    grid = [[0] * w for _ in range(h)]
    for c in (2, 4, 6, 8):          # racks
        for r in range(1, h - 2):
            grid[r][c] = 1
    return grid, (0, 0), 5


def _layout_bottleneck() -> Tuple[List[List[int]], Cell, int]:
    h, w = 11, 11
    grid = [[0] * w for _ in range(h)]
    for c in range(w):              # dividing wall with a single door on the right
        grid[5][c] = 1
    grid[5][9] = 0
    return grid, (0, 0), 5


_LAYOUTS = {
    "open_floor": _layout_open_floor,
    "aisle_racks": _layout_aisle_racks,
    "bottleneck": _layout_bottleneck,
}


def _candidate_cells(family: str, grid, depot: Cell) -> List[Cell]:
    h, w = len(grid), len(grid[0])
    reachable = bfs_distances(tuple(tuple(r) for r in grid), depot)
    cells = [c for c in reachable if c != depot]
    if family == "aisle_racks":
        # pick faces only: floor cells that touch a rack
        cells = [
            (r, c) for (r, c) in cells
            if any(
                0 <= r + dr < h and 0 <= c + dc < w and grid[r + dr][c + dc] == 1
                for dr, dc in MOVES.values()
            )
        ]
    return sorted(cells)


@functools.lru_cache(maxsize=None)
def make_variant(family: str, seed: int) -> Variant:
    if family not in _LAYOUTS:
        raise KeyError(f"unknown family {family!r}")
    grid_list, depot, n_items = _LAYOUTS[family]()
    grid = tuple(tuple(row) for row in grid_list)
    rng = random.Random(f"{family}|{seed}")
    candidates = _candidate_cells(family, grid_list, depot)

    if family == "bottleneck":
        # force items on both sides of the wall so the door must be used
        top = [c for c in candidates if c[0] < 5 and c[0] >= 2]
        bottom = [c for c in candidates if c[0] > 5 and c[1] <= 6]
        items = rng.sample(top, 2) + rng.sample(bottom, 3)
    else:
        items = rng.sample(candidates, n_items)
    items = tuple(sorted(items))

    nodes = (depot,) + items
    dist = pairwise_distances(grid, nodes)
    tour_cost, _ = held_karp(dist)
    optimal_actions = tour_cost + len(items)          # travel + one PICK per item
    slack = max(1, math.ceil(0.10 * optimal_actions))
    budget = optimal_actions + slack

    reach = bfs_distances(grid, depot)
    longest = max(max(row) for row in dist)
    assert longest <= MAX_ALLOWED_DISTANCE, f"{family}:{seed} too large ({longest})"
    assert all(item in reach for item in items)

    return Variant(
        vid=f"{family}:{seed}",
        family=family,
        seed=seed,
        grid=grid,
        depot=depot,
        items=items,
        budget=budget,
        optimal_actions=optimal_actions,
    )


def variant_from_id(vid: str) -> Variant:
    family, _, seed = vid.partition(":")
    return make_variant(family, int(seed))


def all_variants() -> List[Variant]:
    return [make_variant(f, s) for f in FAMILIES for s in SEEDS]


if __name__ == "__main__":
    for v in all_variants():
        print(f"{v.vid:<16} items={v.n_items} optimal={v.optimal_actions:<3} budget={v.budget}")
    print()
    demo = make_variant("bottleneck", 0)
    print(demo.render(pos=demo.depot))
