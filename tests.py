"""Test suite. Runs under pytest, or standalone: `python3 test_suite.py`.

Groups
  A. verifier contract     initial state -> 0, correct solution -> 1,
                           invalid action -> 0, hand-written goal state -> 0,
                           all items picked but not back at depot -> 0
  B. environment sanity    variants solvable, dynamics independent of reward mode
  C. the exploit           the naive reward is satisfied by a wrong solution
  D. the fix works         same trajectories fail the fixed reward, closed loops
                           are free, no false positive in ~45k random rollouts,
                           and a proven reward gap separates solved from unsolved
"""

from __future__ import annotations

import random

from baseline import exact_agent, nearest_neighbour_agent
from env import (
    GOAL_BONUS,
    PICK_CREDIT,
    SHAPING_SCALE,
    STEP_COST,
    OrderPickingEnv,
    rollout,
    success_threshold,
)
from exploit import EXPLOITS, exploit_pick_spam, exploit_shaping_loop, theoretical_bound
from variants import (
    EAST,
    FAMILIES,
    MOVES,
    PICK,
    all_variants,
    bfs_distances,
    variant_from_id,
)
from verifier import verify

VARIANTS = all_variants()


# ------------------------------------------------------------ A. verifier
def test_initial_state_scores_zero():
    """An untouched episode (reset only, no actions) must score 0."""
    for v in VARIANTS:
        env = OrderPickingEnv(v)
        env.reset()
        score, reason = verify(env.final_state(), explain=True)
        assert score == 0, f"{v.vid}: initial state scored 1"
        assert "picked" in reason


def test_correct_solution_scores_one():
    """The exact planner's solution must score 1 on every variant."""
    for v in VARIANTS:
        final_state, _, _ = rollout(v, exact_agent(v))
        score, reason = verify(final_state, explain=True)
        assert score == 1, f"{v.vid}: correct solution scored 0 ({reason})"


def test_invalid_action_scores_zero():
    """Illegal moves, out-of-range action ids and over-budget logs all score 0."""
    for v in VARIANTS:
        good = exact_agent(v)

        # (a) walk off the grid: from the depot at (0,0), NORTH is impossible
        assert verify({"variant_id": v.vid, "actions": [0] + good}) == 0

        # (b) an action id that does not exist
        assert verify({"variant_id": v.vid, "actions": good + [99]}) == 0
        assert verify({"variant_id": v.vid, "actions": good + ["PICK"]}) == 0

        # (c) PICK where there is no item
        assert verify({"variant_id": v.vid, "actions": [PICK] + good}) == 0

        # (d) a correct solution padded past the budget
        padded = good + [EAST, 2] * v.budget
        assert len(padded) > v.budget
        assert verify({"variant_id": v.vid, "actions": padded}) == 0

        # (e) walking into a rack, where the layout has one
        if v.family == "aisle_racks":
            blocked = [(r, c) for r in range(v.height) for c in range(v.width) if v.grid[r][c] == 1]
            assert blocked, "aisle layout should contain racks"


def test_written_goal_state_scores_zero():
    """Declaring victory in the submission fields does not produce a score."""
    for v in VARIANTS:
        # a fully "solved-looking" record with no work behind it
        forged = {
            "variant_id": v.vid,
            "actions": [],
            "claimed_position": v.depot,
            "claimed_picked": tuple(True for _ in v.items),
            "claimed_steps": v.optimal_actions,
            "success": True,
            "reward": 999.0,
            "verified": 1,
        }
        assert verify(forged) == 0, f"{v.vid}: forged goal state scored 1"

        # teleporting: the items are marked picked without ever being reached
        teleport = {
            "variant_id": v.vid,
            "actions": [PICK] * v.n_items,
            "claimed_picked": tuple(True for _ in v.items),
        }
        assert verify(teleport) == 0

        # a valid solution to a *different* variant does not transfer
        other = VARIANTS[(VARIANTS.index(v) + 1) % len(VARIANTS)]
        cross = {"variant_id": v.vid, "actions": exact_agent(other)}
        if other.vid != v.vid:
            assert verify(cross) == 0

        # missing or malformed submissions
        assert verify({}) == 0
        assert verify({"variant_id": v.vid}) == 0
        assert verify({"variant_id": "no_such_family:3", "actions": []}) == 0
        assert verify("solved") == 0


def test_all_picked_but_not_home_scores_zero():
    """Collecting every item is not enough: the picker must end at the depot."""
    for v in VARIANTS:
        solution = exact_agent(v)
        # cut the solution right after the last PICK: every item collected,
        # picker still standing on the last item, not at the depot
        last_pick = max(i for i, a in enumerate(solution) if a == PICK)
        stranded = solution[: last_pick + 1]
        score, reason = verify({"variant_id": v.vid, "actions": stranded}, explain=True)
        assert score == 0, f"{v.vid}: stranded picker scored 1"
        assert "depot" in reason, reason


# --------------------------------------------------------- B. environment
def test_variants_are_generated_and_solvable():
    assert len(FAMILIES) >= 3
    assert len(VARIANTS) >= 3
    assert len({v.vid for v in VARIANTS}) == len(VARIANTS)
    for v in VARIANTS:
        assert v.optimal_actions <= v.budget
        assert len(exact_agent(v)) == v.optimal_actions, f"{v.vid}: optimum mismatch"
        assert variant_from_id(v.vid) == v, "generation must be deterministic"
        assert all(v.is_floor(item) and item != v.depot for item in v.items)


def test_reward_mode_does_not_change_dynamics():
    for v in VARIANTS:
        actions = nearest_neighbour_agent(v)
        naive_state, _, _ = rollout(v, actions, reward_mode="naive")
        fixed_state, _, _ = rollout(v, actions, reward_mode="fixed")
        assert naive_state["actions"] == fixed_state["actions"]
        assert naive_state["claimed_picked"] == fixed_state["claimed_picked"]
        assert verify(naive_state) == verify(fixed_state)


def test_illegal_action_ends_episode_as_invalid():
    v = VARIANTS[0]
    env = OrderPickingEnv(v)
    env.reset()
    _, reward, terminated, truncated, info = env.step(0)  # NORTH out of the grid
    assert info["invalid"] and terminated and reward == 0.0
    assert verify(env.final_state()) == 0


# --------------------------------------------------------- C. the exploit
def test_naive_reward_is_exploitable():
    """A wrong solution reaches the naive success threshold on every variant."""
    hits = 0
    for v in VARIANTS:
        actions = exploit_pick_spam(v)
        final_state, naive_return, _ = rollout(v, actions, reward_mode="naive")
        threshold = success_threshold(v)
        honest_state, honest_return, _ = rollout(v, exact_agent(v), reward_mode="naive")

        assert verify(final_state) == 0, f"{v.vid}: exploit should not solve the task"
        assert naive_return >= threshold, f"{v.vid}: exploit return {naive_return} < {threshold}"
        assert naive_return > honest_return, "the exploit should even beat an honest optimal run"
        hits += 1
    assert hits == len(VARIANTS)


def test_naive_reward_pays_for_closed_loops():
    """Under the naive reward, going nowhere earns money."""
    for v in VARIANTS:
        _, naive_return, _ = rollout(v, exploit_shaping_loop(v), reward_mode="naive")
        # strip the step cost: the shaping bonus alone is strictly positive
        loop_len = len(exploit_shaping_loop(v))
        shaping_only = naive_return + STEP_COST * loop_len
        assert shaping_only > 0, f"{v.vid}: expected positive loop payoff"


# ---------------------------------------------------------- D. the fix works
def test_fix_blocks_the_exploit():
    """The very same exploit trajectories fall far below threshold once fixed."""
    for v in VARIANTS:
        for name, build in EXPLOITS.items():
            actions = build(v)
            final_state, fixed_return, _ = rollout(v, actions, reward_mode="fixed")
            assert verify(final_state) == 0
            assert fixed_return < success_threshold(v), f"{v.vid}/{name} still passes"


def test_fixed_shaping_is_free_on_closed_loops():
    """Potential-based shaping: any loop back to the same state nets exactly 0."""
    rng = random.Random(7)
    for v in VARIANTS:
        env = OrderPickingEnv(v, reward_mode="fixed")
        env.reset()
        start_pos, start_phi = env.state.pos, env.potential()
        trail, shaping_sum = [], 0.0
        for _ in range(12):  # random walk out
            legal = [a for a, (dr, dc) in MOVES.items() if v.is_floor((env.state.pos[0] + dr, env.state.pos[1] + dc))]
            a = rng.choice(legal)
            _, r, term, trunc, _ = env.step(a)
            shaping_sum += r + STEP_COST  # remove the step cost, keep the shaping
            trail.append(a)
            if term or trunc:
                break
        for a in reversed(trail):  # exact retrace back to the start
            _, r, term, trunc, _ = env.step({0: 1, 1: 0, 2: 3, 3: 2}[a])
            shaping_sum += r + STEP_COST
            if term or trunc:
                break
        if env.state.pos == start_pos and not env.state.invalid:
            assert abs(shaping_sum) < 1e-9, f"{v.vid}: closed loop paid {shaping_sum}"
            assert abs(env.potential() - start_phi) < 1e-9


def test_fixed_reward_has_no_false_positives_under_random_play():
    """~45k legal random episodes: fixed return >= threshold implies verifier == 1."""
    rng = random.Random(1234)
    checked = naive_false_positives = fixed_false_positives = 0
    for v in VARIANTS:
        for _ in range(3000):
            env_fixed = OrderPickingEnv(v, reward_mode="fixed")
            env_naive = OrderPickingEnv(v, reward_mode="naive")
            env_fixed.reset()
            env_naive.reset()
            fixed_return = naive_return = 0.0
            while not env_fixed.state.done:
                pos = env_fixed.state.pos
                legal = [a for a, (dr, dc) in MOVES.items() if v.is_floor((pos[0] + dr, pos[1] + dc))]
                if pos in v.items:
                    legal += [PICK, PICK, PICK]  # bias toward picking
                a = rng.choice(legal)
                _, rf, _, _, _ = env_fixed.step(a)
                _, rn, _, _, _ = env_naive.step(a)
                fixed_return += rf
                naive_return += rn
            solved = verify(env_fixed.final_state())
            threshold = success_threshold(v)
            checked += 1
            if fixed_return >= threshold and not solved:
                fixed_false_positives += 1
            if naive_return >= threshold and not solved:
                naive_false_positives += 1
            assert not (fixed_return >= threshold and not solved), (
                f"{v.vid}: fixed reward {fixed_return:.2f} passed an unsolved episode"
            )
    assert checked >= 45000
    assert fixed_false_positives == 0


def test_fixed_reward_gap_is_proven_not_just_observed():
    """Arithmetic separation: any unsolved episode < threshold <= any solved one."""
    for v in VARIANTS:
        b = theoretical_bound(v)
        d0 = min(bfs_distances(v.grid, v.depot)[c] for c in v.items)
        # upper bound for a non-solving episode
        assert b["max_non_solving_return"] == (v.n_items - 1) * PICK_CREDIT + SHAPING_SCALE * d0
        assert b["max_non_solving_return"] < b["threshold"], f"{v.vid}: no headroom"
        # lower bound for a solving episode
        assert b["min_solving_return"] == v.n_items * PICK_CREDIT + GOAL_BONUS - STEP_COST * v.budget
        assert b["min_solving_return"] > b["threshold"], f"{v.vid}: solving run could fail"


def test_fixed_reward_prefers_shorter_routes():
    """Policy quality is still visible: a detour is worth strictly less."""
    compared = 0
    for v in VARIANTS:
        opt_actions = exact_agent(v)
        nn_actions = nearest_neighbour_agent(v)
        _, opt_return, _ = rollout(v, opt_actions, reward_mode="fixed")
        nn_state, nn_return, _ = rollout(v, nn_actions, reward_mode="fixed")
        if verify(nn_state) == 1 and len(nn_actions) > len(opt_actions):
            assert nn_return < opt_return
            compared += 1
    assert compared >= 1, "expected at least one suboptimal-but-valid baseline run"


# ------------------------------------------------------------------ runner
def main() -> int:
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures.append((fn.__name__, exc))
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # a crash is a failure too, not a reason to stop
            failures.append((fn.__name__, exc))
            print(f"FAIL  {fn.__name__}: crashed with {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
