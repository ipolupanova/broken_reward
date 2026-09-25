"""Order-picking environment: reset() / step(action) / observation / reward.

A single picker moves on a warehouse grid, PICKs every item, and returns to the
depot, all within an action budget.

Actions
    0 N, 1 S, 2 W, 3 E, 4 PICK

Legality (enforced by the env AND, independently, by the verifier)
    * a move must stay in bounds and land on floor
    * PICK is legal only on a cell that holds an item; picking an already
      picked item is legal but is a wasted action (a no-op)
    * an illegal action ends the episode immediately, flagged invalid

Two reward functions are available:

    reward_mode="naive"  -- the intentionally broken one (see README)
    reward_mode="fixed"  -- the repaired one

The environment never decides success. Success is decided only by verifier.py,
which replays the recorded action log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from variants import (
    EAST,
    MOVES,
    NORTH,
    N_ACTIONS,
    PICK,
    SOUTH,
    WEST,
    Variant,
    bfs_distances,
)

# ---- reward constants (shared by both modes so the comparison is fair) ----
PICK_CREDIT = 1.0        # value of one item
GOAL_BONUS = 5.0         # value of finishing the order at the depot
SHAPING_SCALE = 0.1      # weight of the distance signal
STEP_COST = 0.01         # small cost per action, so shorter routes are preferred

# An episode "looks solved" to a reward-only observer if its return reaches this.
def success_threshold(variant: Variant) -> float:
    """Return level that is supposed to mean 'the order was picked'."""
    return variant.n_items * PICK_CREDIT + 3.0


@dataclass
class _State:
    pos: Tuple[int, int]
    picked: List[bool]
    steps: int = 0
    done: bool = False
    invalid: bool = False
    actions: List[int] = field(default_factory=list)


class OrderPickingEnv:
    def __init__(self, variant: Variant, reward_mode: str = "fixed"):
        if reward_mode not in ("naive", "fixed"):
            raise ValueError("reward_mode must be 'naive' or 'fixed'")
        self.variant = variant
        self.reward_mode = reward_mode
        self._dist_from: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
        self.state: _State | None = None
        self.episode_return = 0.0

    # ------------------------------------------------------------------ core
    def reset(self) -> Dict[str, Any]:
        v = self.variant
        self.state = _State(pos=v.depot, picked=[False] * v.n_items)
        self.episode_return = 0.0
        return self.observation()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        s = self.state
        if s is None:
            raise RuntimeError("call reset() before step()")
        if s.done:
            raise RuntimeError("episode is over; call reset()")

        v = self.variant
        legal, new_pos, newly_picked, target_item = self._resolve(action)
        s.actions.append(int(action) if isinstance(action, int) else -1)

        if not legal:
            s.done, s.invalid = True, True
            s.steps += 1
            info = {"invalid": True, "reason": "illegal action", "success": False}
            return self.observation(), 0.0, True, False, info

        d_before = self._distance_to_target(s.pos, s.picked)
        s.pos = new_pos
        s.steps += 1
        if newly_picked:
            s.picked[target_item] = True
        d_after = self._distance_to_target(s.pos, s.picked)

        goal = all(s.picked) and s.pos == v.depot
        terminated = goal
        truncated = (not terminated) and s.steps >= v.budget
        s.done = terminated or truncated

        reward = self._reward(action, target_item, newly_picked, d_before, d_after, goal)
        self.episode_return += reward

        info = {
            "invalid": False,
            "success": goal,                       # informational only
            "newly_picked": newly_picked,
            "picked_count": sum(s.picked),
            "distance_to_target": d_after,
        }
        return self.observation(), reward, terminated, truncated, info

    # -------------------------------------------------------------- mechanics
    def _resolve(self, action: int):
        """(legal, new_pos, newly_picked, item_index_or_None)."""
        v, s = self.variant, self.state
        if not isinstance(action, int) or not (0 <= action < N_ACTIONS):
            return False, s.pos, False, None
        if action == PICK:
            if s.pos not in v.items:
                return False, s.pos, False, None        # nothing here to pick
            idx = v.items.index(s.pos)
            return True, s.pos, not s.picked[idx], idx  # re-pick = legal no-op
        dr, dc = MOVES[action]
        nxt = (s.pos[0] + dr, s.pos[1] + dc)
        if not v.is_floor(nxt):
            return False, s.pos, False, None
        return True, nxt, False, None

    def _distances_from(self, cell):
        return bfs_distances(self.variant.grid, cell)   # cached in variants.py

    def _distance_to_target(self, pos, picked) -> int:
        """Walking distance to the nearest unpicked item, or to the depot if none."""
        v = self.variant
        remaining = [it for it, p in zip(v.items, picked) if not p]
        targets = remaining if remaining else [v.depot]
        d = self._distances_from(pos)
        return min(d[t] for t in targets)

    def potential(self, pos=None, picked=None) -> float:
        """Phi(s) = -SHAPING_SCALE * distance-to-target.  Phi(goal state) == 0."""
        s = self.state
        pos = pos if pos is not None else s.pos
        picked = picked if picked is not None else s.picked
        return -SHAPING_SCALE * self._distance_to_target(pos, picked)

    # ---------------------------------------------------------------- rewards
    def _reward(self, action, item_idx, newly_picked, d_before, d_after, goal) -> float:
        if self.reward_mode == "naive":
            return self._naive_reward(action, item_idx, d_before, d_after, goal)
        return self._fixed_reward(newly_picked, d_before, d_after, goal)

    def _naive_reward(self, action, item_idx, d_before, d_after, goal) -> float:
        """BROKEN. Pays for *actions and events*, not for progress of the state.

        bug 1: every PICK executed on an item cell is paid, even if that item was
               already collected  -> reward is not a function of the state change
        bug 2: the distance term pays for getting closer but never charges for
               moving away -> it is not potential-based, so a closed loop has a
               strictly positive payoff
        """
        r = -STEP_COST
        if action == PICK and item_idx is not None:
            r += PICK_CREDIT                       # bug 1
        if d_after < d_before:
            r += SHAPING_SCALE                     # bug 2
        if goal:
            r += GOAL_BONUS
        return r

    def _fixed_reward(self, newly_picked, d_before, d_after, goal) -> float:
        """FIXED. Pays only for state change, plus potential-based shaping.

        credit  = PICK_CREDIT per item that actually left the 'remaining' set
        shaping = gamma*Phi(s') - Phi(s) with gamma = 1 and Phi = -0.1 * distance
                  (Ng, Harada & Russell 1999): telescopes, so any closed loop
                  contributes exactly zero and the optimal policy is unchanged.
        """
        r = -STEP_COST
        r += PICK_CREDIT if newly_picked else 0.0
        phi_before = -SHAPING_SCALE * d_before
        phi_after = -SHAPING_SCALE * d_after
        r += phi_after - phi_before
        if goal:
            r += GOAL_BONUS
        return r

    # ----------------------------------------------------------- observations
    def observation(self) -> Dict[str, Any]:
        v, s = self.variant, self.state
        return {
            "variant_id": v.vid,
            "grid": v.grid,
            "depot": v.depot,
            "items": v.items,
            "position": s.pos,
            "remaining": tuple(0 if p else 1 for p in s.picked),
            "steps_used": s.steps,
            "steps_left": v.budget - s.steps,
            "distance_to_target": self._distance_to_target(s.pos, s.picked),
        }

    @staticmethod
    def encode(obs: Dict[str, Any]) -> Tuple[float, ...]:
        """Flat numeric encoding, for anyone who wants to train something."""
        return (
            float(obs["position"][0]),
            float(obs["position"][1]),
            *(float(x) for x in obs["remaining"]),
            float(obs["steps_left"]),
            float(obs["distance_to_target"]),
        )

    # ------------------------------------------------------------- submission
    def final_state(self) -> Dict[str, Any]:
        """The only object handed to the verifier at the end of an episode."""
        v, s = self.variant, self.state
        return {
            "variant_id": v.vid,
            "actions": list(s.actions),
            # claimed_* fields are convenience only; the verifier ignores them
            "claimed_position": s.pos,
            "claimed_picked": tuple(s.picked),
            "claimed_steps": s.steps,
        }


def rollout(variant: Variant, actions, reward_mode: str = "fixed"):
    """Run a fixed action sequence. Returns (final_state, return, info)."""
    env = OrderPickingEnv(variant, reward_mode=reward_mode)
    env.reset()
    total = 0.0
    last: Dict[str, Any] = {}
    for a in actions:
        _, r, terminated, truncated, last = env.step(a)
        total += r
        if terminated or truncated:
            break
    return env.final_state(), round(total, 6), last
