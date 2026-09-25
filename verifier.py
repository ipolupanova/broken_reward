"""Automatic success checker. No human, no reward, no trust.

The verifier receives one object: the final state of an episode. From it, it
reads exactly two fields:

    variant_id   -- rebuilds the task instance itself (generation is deterministic)
    actions      -- the recorded action log

Everything else in the submission (claimed_position, claimed_picked, a whole
hand-written "goal state", ...) is ignored. The verifier re-simulates the
dynamics with its own implementation -- deliberately not importing env.py -- so
a bug or a cheat in the environment cannot make a wrong episode pass.

verify() returns 1 (solved) or 0 (not solved). Nothing in between.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from variants import MOVES, N_ACTIONS, PICK, Variant, variant_from_id


def verify(final_state: Dict[str, Any], explain: bool = False):
    """Score a submission: 1 if the order was legally picked, else 0."""
    reason = "ok"
    score = 0

    if not isinstance(final_state, dict):
        reason = "submission is not a record"
        return (0, reason) if explain else 0

    vid = final_state.get("variant_id")
    actions = final_state.get("actions")

    if not isinstance(vid, str):
        reason = "missing variant_id"
        return (0, reason) if explain else 0
    try:
        variant = variant_from_id(vid)
    except Exception:
        reason = f"unknown variant_id {vid!r}"
        return (0, reason) if explain else 0

    if not isinstance(actions, (list, tuple)):
        reason = "missing action log"
        return (0, reason) if explain else 0
    if len(actions) > variant.budget:
        reason = f"over budget ({len(actions)} > {variant.budget})"
        return (0, reason) if explain else 0

    ok, reason, pos, picked = _replay(variant, list(actions))
    if not ok:
        return (0, reason) if explain else 0

    if not all(picked):
        reason = f"only {sum(picked)}/{variant.n_items} items picked"
    elif pos != variant.depot:
        reason = f"picker ended at {pos}, not at depot {variant.depot}"
    else:
        score, reason = 1, "solved"

    return (score, reason) if explain else score


def _replay(variant: Variant, actions: List[int]) -> Tuple[bool, str, Tuple[int, int], List[bool]]:
    """Independent re-simulation. Returns (legal, reason, final_pos, picked)."""
    pos = variant.depot
    picked = [False] * variant.n_items
    for t, action in enumerate(actions):
        if not isinstance(action, int) or isinstance(action, bool) or not (0 <= action < N_ACTIONS):
            return False, f"action {t}: {action!r} is not a valid action id", pos, picked
        if action == PICK:
            if pos not in variant.items:
                return False, f"action {t}: PICK at {pos} where there is no item", pos, picked
            picked[variant.items.index(pos)] = True
            continue
        dr, dc = MOVES[action]
        nxt = (pos[0] + dr, pos[1] + dc)
        if not variant.in_bounds(nxt):
            return False, f"action {t}: move off the grid to {nxt}", pos, picked
        if variant.grid[nxt[0]][nxt[1]] != 0:
            return False, f"action {t}: move into a rack at {nxt}", pos, picked
        pos = nxt
    return True, "ok", pos, picked


def verify_all(submissions) -> Dict[str, int]:
    return {sub.get("variant_id", f"#{i}"): verify(sub) for i, sub in enumerate(submissions)}
