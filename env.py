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

PICK_CREDIT = 1.0 #per item picked
GOAL_BONUS = 5.0 #complete order back at the desk
SHAPING_SCALE = 0.1 #per step closer to the next item
STEP_COST = 0.01 #per move

def success_threshold(variant: Variant) -> float:
    #points that count as a good shift
    return variant.n_items * PICK_CREDIT + 3.0


@dataclass
class _State:
    #what the picker has done so far
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

        #an illegal action ends the shift
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

        goal = all(s.picked) and s.pos == v.depot #all picked and back at the desk
        terminated = goal
        truncated = (not terminated) and s.steps >= v.budget #budget used up
        s.done = terminated or truncated

        reward = self._reward(action, target_item, newly_picked, d_before, d_after, goal)
        self.episode_return += reward

        info = {
            "invalid": False,
            "success": goal,
            "newly_picked": newly_picked,
            "picked_count": sum(s.picked),
            "distance_to_target": d_after,
        }
        return self.observation(), reward, terminated, truncated, info

    def _resolve(self, action: int):
        #checks if the action is legal and what it changes
        v, s = self.variant, self.state
        if not isinstance(action, int) or not (0 <= action < N_ACTIONS):
            return False, s.pos, False, None
        if action == PICK:
            if s.pos not in v.items:
                return False, s.pos, False, None
            idx = v.items.index(s.pos)
            return True, s.pos, not s.picked[idx], idx
        dr, dc = MOVES[action]
        nxt = (s.pos[0] + dr, s.pos[1] + dc)
        if not v.is_floor(nxt):
            return False, s.pos, False, None
        return True, nxt, False, None

    def _distances_from(self, cell):
        return bfs_distances(self.variant.grid, cell)

    def _distance_to_target(self, pos, picked) -> int:
        #distance to the nearest remaining item, or to the desk once all are picked
        v = self.variant
        remaining = [it for it, p in zip(v.items, picked) if not p]
        targets = remaining if remaining else [v.depot]
        d = self._distances_from(pos)
        return min(d[t] for t in targets)

    def potential(self, pos=None, picked=None) -> float:
        s = self.state
        pos = pos if pos is not None else s.pos
        picked = picked if picked is not None else s.picked
        return -SHAPING_SCALE * self._distance_to_target(pos, picked)

    def _reward(self, action, item_idx, newly_picked, d_before, d_after, goal) -> float:
        if self.reward_mode == "naive":
            return self._naive_reward(action, item_idx, d_before, d_after, goal)
        return self._fixed_reward(newly_picked, d_before, d_after, goal)

    def _naive_reward(self, action, item_idx, d_before, d_after, goal) -> float:
        #broken reward
        r = -STEP_COST
        if action == PICK and item_idx is not None:
            r += PICK_CREDIT #paid for every pick, even of an item already picked
        if d_after < d_before:
            r += SHAPING_SCALE #paid for closer, never charged for away
        if goal:
            r += GOAL_BONUS
        return r

    def _fixed_reward(self, newly_picked, d_before, d_after, goal) -> float:
        #fixed reward
        r = -STEP_COST
        r += PICK_CREDIT if newly_picked else 0.0 #paid only the first time an item is picked
        phi_before = -SHAPING_SCALE * d_before
        phi_after = -SHAPING_SCALE * d_after
        r += phi_after - phi_before #works both ways, a loop earns 0
        if goal:
            r += GOAL_BONUS
        return r

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
        #flat vector, for training
        return (
            float(obs["position"][0]),
            float(obs["position"][1]),
            *(float(x) for x in obs["remaining"]),
            float(obs["steps_left"]),
            float(obs["distance_to_target"]),
        )

    def final_state(self) -> Dict[str, Any]:
        #what the picker hands to the verifier
        v, s = self.variant, self.state
        return {
            "variant_id": v.vid,
            "actions": list(s.actions),
            "claimed_position": s.pos, #claimed_* is ignored by the verifier
            "claimed_picked": tuple(s.picked),
            "claimed_steps": s.steps,
        }


def rollout(variant: Variant, actions, reward_mode: str = "fixed"):
    #runs a fixed list of actions
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
