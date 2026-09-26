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


#verifier
def test_initial_state_scores_zero():
    for v in VARIANTS:
        env = OrderPickingEnv(v)
        env.reset()
        score, reason = verify(env.final_state(), explain=True)
        assert score == 0, f"{v.vid}: initial state scored 1"
        assert "picked" in reason


def test_correct_solution_scores_one():
    for v in VARIANTS:
        final_state, _, _ = rollout(v, exact_agent(v))
        score, reason = verify(final_state, explain=True)
        assert score == 1, f"{v.vid}: correct solution scored 0 ({reason})"


def test_invalid_action_scores_zero():
    for v in VARIANTS:
        good = exact_agent(v)

        assert verify({"variant_id": v.vid, "actions": [0] + good}) == 0 #off the grid

        assert verify({"variant_id": v.vid, "actions": good + [99]}) == 0 #unknown action id
        assert verify({"variant_id": v.vid, "actions": good + ["PICK"]}) == 0

        assert verify({"variant_id": v.vid, "actions": [PICK] + good}) == 0 #pick with no item

        padded = good + [EAST, 2] * v.budget #over budget
        assert len(padded) > v.budget
        assert verify({"variant_id": v.vid, "actions": padded}) == 0

        if v.family == "aisle_racks":
            blocked = [(r, c) for r in range(v.height) for c in range(v.width) if v.grid[r][c] == 1]
            assert blocked, "aisle layout should contain racks"


def test_written_goal_state_scores_zero():
    for v in VARIANTS:
        #"I'm done" record with no actions behind it
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

        teleport = {
            "variant_id": v.vid,
            "actions": [PICK] * v.n_items,
            "claimed_picked": tuple(True for _ in v.items),
        }
        assert verify(teleport) == 0

        #solution copied from another room
        other = VARIANTS[(VARIANTS.index(v) + 1) % len(VARIANTS)]
        cross = {"variant_id": v.vid, "actions": exact_agent(other)}
        if other.vid != v.vid:
            assert verify(cross) == 0

        assert verify({}) == 0
        assert verify({"variant_id": v.vid}) == 0
        assert verify({"variant_id": "no_such_family:3", "actions": []}) == 0
        assert verify("solved") == 0


def test_all_picked_but_not_home_scores_zero():
    for v in VARIANTS:
        solution = exact_agent(v)
        last_pick = max(i for i, a in enumerate(solution) if a == PICK)
        stranded = solution[: last_pick + 1]
        score, reason = verify({"variant_id": v.vid, "actions": stranded}, explain=True)
        assert score == 0, f"{v.vid}: stranded picker scored 1"
        assert "depot" in reason, reason


#environment
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
    _, reward, terminated, truncated, info = env.step(0)
    assert info["invalid"] and terminated and reward == 0.0
    assert verify(env.final_state()) == 0


#exploit
def test_naive_reward_is_exploitable():
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
    for v in VARIANTS:
        _, naive_return, _ = rollout(v, exploit_shaping_loop(v), reward_mode="naive")
        loop_len = len(exploit_shaping_loop(v))
        shaping_only = naive_return + STEP_COST * loop_len
        assert shaping_only > 0, f"{v.vid}: expected positive loop payoff"


#fix
def test_fix_blocks_the_exploit():
    for v in VARIANTS:
        for name, build in EXPLOITS.items():
            actions = build(v)
            final_state, fixed_return, _ = rollout(v, actions, reward_mode="fixed")
            assert verify(final_state) == 0
            assert fixed_return < success_threshold(v), f"{v.vid}/{name} still passes"


def test_fixed_shaping_is_free_on_closed_loops():
    rng = random.Random(7)
    for v in VARIANTS:
        env = OrderPickingEnv(v, reward_mode="fixed")
        env.reset()
        start_pos, start_phi = env.state.pos, env.potential()
        trail, shaping_sum = [], 0.0
        for _ in range(12): #walk out
            legal = [a for a, (dr, dc) in MOVES.items() if v.is_floor((env.state.pos[0] + dr, env.state.pos[1] + dc))]
            a = rng.choice(legal)
            _, r, term, trunc, _ = env.step(a)
            shaping_sum += r + STEP_COST
            trail.append(a)
            if term or trunc:
                break
        for a in reversed(trail): #walk back the same way
            _, r, term, trunc, _ = env.step({0: 1, 1: 0, 2: 3, 3: 2}[a])
            shaping_sum += r + STEP_COST
            if term or trunc:
                break
        if env.state.pos == start_pos and not env.state.invalid:
            assert abs(shaping_sum) < 1e-9, f"{v.vid}: closed loop paid {shaping_sum}"
            assert abs(env.potential() - start_phi) < 1e-9


def test_fixed_reward_has_no_false_positives_under_random_play():
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
                    legal += [PICK, PICK, PICK] #bias toward picking
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
    for v in VARIANTS:
        b = theoretical_bound(v)
        d0 = min(bfs_distances(v.grid, v.depot)[c] for c in v.items)
        assert b["max_non_solving_return"] == v.n_items * PICK_CREDIT + SHAPING_SCALE * d0 - SHAPING_SCALE
        assert b["max_non_solving_return"] < b["threshold"], f"{v.vid}: no headroom"
        #all items picked, one step short of the desk
        stranded_state, stranded_return, _ = rollout(v, exact_agent(v)[:-1], reward_mode="fixed")
        assert verify(stranded_state) == 0
        assert stranded_return <= b["max_non_solving_return"], f"{v.vid}: stranded run beats the bound"
        assert b["min_solving_return"] == v.n_items * PICK_CREDIT + GOAL_BONUS - STEP_COST * v.budget
        assert b["min_solving_return"] > b["threshold"], f"{v.vid}: solving run could fail"


def test_fixed_reward_prefers_shorter_routes():
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


def main() -> int:
    #runs all tests without pytest
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures.append((fn.__name__, exc))
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:
            failures.append((fn.__name__, exc))
            print(f"FAIL  {fn.__name__}: crashed with {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
