"""Training-free baseline agents.

nearest_neighbour_agent : greedy heuristic, repeatedly walks to the closest
                          unpicked item, then returns to the depot
exact_agent             : Held-Karp optimal tour (used as the reference solver
                          and by the verifier tests)

Both emit an action sequence; success is always judged by verifier.verify().
"""

from __future__ import annotations

from typing import Callable, Dict, List

from env import rollout, success_threshold
from variants import (
    PICK,
    Variant,
    all_variants,
    bfs_distances,
    bfs_move_path,
    held_karp,
    pairwise_distances,
)
from verifier import verify


def nearest_neighbour_agent(variant: Variant) -> List[int]:
    pos = variant.depot
    remaining = list(variant.items)
    actions: List[int] = []
    while remaining:
        dist = bfs_distances(variant.grid, pos)
        target = min(remaining, key=lambda cell: (dist[cell], cell))
        actions += bfs_move_path(variant.grid, pos, target)
        actions.append(PICK)
        pos = target
        remaining.remove(target)
    actions += bfs_move_path(variant.grid, pos, variant.depot)
    return actions


def exact_agent(variant: Variant) -> List[int]:
    nodes = (variant.depot,) + variant.items
    dist = pairwise_distances(variant.grid, nodes)
    _, order = held_karp(dist)
    pos = variant.depot
    actions: List[int] = []
    for idx in order:
        target = variant.items[idx]
        actions += bfs_move_path(variant.grid, pos, target)
        actions.append(PICK)
        pos = target
    actions += bfs_move_path(variant.grid, pos, variant.depot)
    return actions


def random_agent(seed: int = 0) -> Callable[[Variant], List[int]]:
    import random

    def agent(variant: Variant) -> List[int]:
        rng = random.Random(f"{variant.vid}|{seed}")
        return [rng.randrange(5) for _ in range(variant.budget)]

    return agent


AGENTS: Dict[str, Callable[[Variant], List[int]]] = {
    "nearest_neighbour": nearest_neighbour_agent,
    "exact_planner": exact_agent,
    "random": random_agent(),
}


def evaluate(agent: Callable[[Variant], List[int]], reward_mode: str = "fixed", variants=None):
    """Run an agent on every variant. Returns (rows, success_rate)."""
    variants = variants if variants is not None else all_variants()
    rows = []
    for v in variants:
        actions = agent(v)
        final_state, total, _ = rollout(v, actions, reward_mode=reward_mode)
        score, reason = verify(final_state, explain=True)
        rows.append(
            {
                "variant": v.vid,
                "family": v.family,
                "actions_used": len(final_state["actions"]),
                "budget": v.budget,
                "optimal": v.optimal_actions,
                "return": total,
                "threshold": success_threshold(v),
                "verified": score,
                "reason": reason,
            }
        )
    rate = sum(r["verified"] for r in rows) / len(rows)
    return rows, rate


def report(reward_mode: str = "fixed") -> None:
    print(f"baseline agents on {len(all_variants())} variants (reward_mode={reward_mode})\n")
    for name, agent in AGENTS.items():
        rows, rate = evaluate(agent, reward_mode=reward_mode)
        print(f"--- {name}: success rate {rate:.0%} ({sum(r['verified'] for r in rows)}/{len(rows)})")
        print(f"{'variant':<16}{'used':>6}{'budget':>8}{'optimal':>9}{'return':>9}{'ok':>4}  reason")
        for r in rows:
            print(
                f"{r['variant']:<16}{r['actions_used']:>6}{r['budget']:>8}{r['optimal']:>9}"
                f"{r['return']:>9.2f}{r['verified']:>4}  {r['reason']}"
            )
        by_family: Dict[str, List[int]] = {}
        for r in rows:
            by_family.setdefault(r["family"], []).append(r["verified"])
        per_family = ", ".join(f"{f} {sum(v)}/{len(v)}" for f, v in by_family.items())
        print(f"    per family: {per_family}\n")


if __name__ == "__main__":
    report()
