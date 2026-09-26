# The night shift

My small RL environment for picking up items from an order in a furniture warehouse. There is a deliberately broken reward, an exploit of it, a fix, and tests proving the fix works. 

## Run it [`run.sh`](run.sh), [`run_all.py`](run_all.py)

```bash
bash run.sh
```

`./run.sh` and `python3 run_all.py` do the same. Needs Python 3.9 or newer.

---


## Files

| file | what's in it |
|---|---|
| [env.py](env.py) | `reset()`, `step()`, the observation, broken and fixed reward |
| [variants.py](variants.py) | builds the rooms and computes the perfect route |
| [baseline.py](baseline.py) | the three baseline pickers |
| [verifier.py](verifier.py) | the verifier |
| [exploit.py](exploit.py) | the two loopholes and the evidence tables |
| [tests.py](tests.py) | 15 tests |
| [run_all.py](run_all.py), [run.sh](run.sh) | run command|


## The setting [`Variant`](variants.py#L24)

We are in a furniture warehouse (the big Swedish one, for example) :)
The only person working is the picker, who walks the storage rooms collecting
the items for an order.

A storage room is a grid, seen from above:

```
D..........      D   desk: the picker starts and ends here
..#.#.#.#..      #   shelving rack
..#x#.#x#..      .   aisle floor
..#.#x#.#..      x   an item
..#.#x#x#..
..#.#.#.#..
..#.#.#.#..
...........
...........
```

---

## The task [`env.py`](env.py), [`OrderPickingEnv`](env.py#L39)

The picker has to pick every item ([`Variant.items`](variants.py#L31)) of the order and bring them back to the
desk ([`Variant.depot`](variants.py#L30)) before the shift ends. The shift is a fixed budget of moves ([`Variant.budget`](variants.py#L32)).

- [`reset()`](env.py#L49): the picker clocks in at the desk with an empty cart ([`_State.picked`](env.py#L32))
- [`step(action)`](env.py#L55): the picker does one action, which costs one move ([`_State.steps`](env.py#L33))
- [`observation()`](env.py#L157): everything the picker knows at this moment, for example could look like: 

```python
obs = {
    "position":           (0, 0),            # where the picker is standing
    "remaining":          (1, 1, 1, 1, 1),   # 1 = item still on the shelf
    "steps_used":         0,
    "steps_left":         41,
    "distance_to_target": 5,                 # walking distance to the nearest remaining item
    # plus the room map, the desk position and the item positions
}
```

The picker can do exactly five things ([action ids](variants.py#L12)):

| action | id | what happens |
|---|---|---|
| step north | `0` | one cell up ([`MOVES`](variants.py#L13)) |
| step south | `1` | one cell down |
| step west | `2` | one cell left |
| step east | `3` | one cell right |
| pick | `4` | put the item he is standing at into the cart ([`env.py`](env.py#L102-L106)) |


### The reward [`_naive_reward()`](env.py#L135)

Management sets up a reward system ([`_naive_reward()`](env.py#L135)) to keep the picker motivated:

- **1 point**  for every pick on an item's cell
- **0.1 points**  for every step that brings the picker closer to the nearest item still on the order ([`_distance_to_target()`](env.py#L116))
- **5 points** for bringing the complete order back to the desk
- **−0.01 points** for every move, so taking a long detour costs something

Management looks only at the points, and counts a shift as a *good shift* if it
earns at least **number of items + 3** ([`success_threshold()`](env.py#L23)) (8 points for an order of five-items). 

## The verifier [`verifier.py`](verifier.py), [`verify()`](verifier.py#L10)

The verifier decides whether an episode actually solved the task. It returns 1 or
0 and never looks at the reward.

It receives the final state of an episode ([`final_state()`](env.py#L182)) and
reads only two things from it: which room was played
([`variant_id`](verifier.py#L20)) and the list of actions taken
([`actions`](verifier.py#L21)).

It then:

1. rebuilds the room from its `variant_id`
   ([`variant_from_id()`](variants.py#L255)), so the room is never taken from
   the episode itself,
2. replays every action from the starting point ([`_replay()`](verifier.py#L53)) and
   returns 0 at the first illegal action, or if there are more actions than the
   budget allows,
3. returns 1 only if every item was picked and the picker ended at the starting point.

| case | score | test |
|---|---|---|
| no actions taken | 0 | [`test_initial_state_scores_zero`](tests.py#L31) |
| the optimal route | 1 | [`test_correct_solution_scores_one`](tests.py#L40) |
| an illegal action, e.g. walking off the grid | 0 | [`test_invalid_action_scores_zero`](tests.py#L47) |
| the final state says the task is done, but replaying its actions shows it wasn't | 0 | [`test_written_goal_state_scores_zero`](tests.py#L67) |
| every item picked, but not back at the starting point | 0 | [`test_all_picked_but_not_home_scores_zero`](tests.py#L101) |

## Three task variants = three kinds of room [`variants.py`](variants.py), [`make_variant()`](variants.py#L214)

Each kind of room is generated with 5 different item layouts, so
there are 15 tasks in total:

| room | size | items | perfect route |
|---|---|---|---|
| `open_floor` | 7 × 7, no racks | 3 | 17–25 moves |
| `aisle_racks` | 9 × 11, four racks | 5 | 35–41 moves |
| `bottleneck` | 11 × 11, two halves joined by one door | 5 | 53–57 moves |

For every room the shortest route is computed exactly (BFS for walking
distances, Held–Karp for the order of items), and the move budget for a shift is that route plus
10 %.

---

## Baseline agents [`baseline.py`](baseline.py), [`evaluate()`](baseline.py#L68)

| picker | strategy | solved |
|---|---|---|
| [`exact_agent`](baseline.py#L34) | follows the perfect route | 15 / 15 |
| [`nearest_neighbour_agent`](baseline.py#L18) | always walks to the closest remaining item | 14 / 15 (runs out of moves in `aisle_racks:1`) |
| [`random_agent`](baseline.py#L50) | random actions | 0 / 15 (always makes an illegal move early) |

[`evaluate()`](baseline.py#L68) runs a picker on all rooms and asks the
verifier. This shows every room is solvable.

```
--- nearest_neighbour: success rate 93% (14/15)
--- exact_planner: success rate 100% (15/15)
--- random: success rate 0% (0/15)
```

`python3 baseline.py` prints every room for every agent and the success rates

---

## The exploit [`exploit.py`](exploit.py)

A sneaky picker reads the reward system and finds two loopholes:

1. **Pick spam**: every pick on an item's cell pays 1 point, even if that item has already been picked. So the picker walks to the nearest item and repeats "pick" until the move budget runs out.
2. **Pacing backand forth**: a step closer to an item pays 0.1, a step away costs nothing. Stepping back and forth earns points without completing the task.

In `aisle_racks:0`, pick spam earns 36.09 points against 12.63 for the perfect route, clearing the good-shift line while the verifier scores it 0. It clears the line in all 15 rooms.


```python
# env.py, _naive_reward()
if action == PICK and item_idx is not None:
    r += PICK_CREDIT          # paid for every pick, even of an item already in the cart
if d_after < d_before:
    r += SHAPING_SCALE        # paid for getting closer, never charged for walking away
```

---

## The fix [`_fixed_reward()`](env.py#L146)

1. An item pays 1 point only the first time it goes into the cart.
2. The distance bonus works both ways: +0.1 per step closer, −0.1 per step away. A walk that ends where it started earns exactly 0. 

```python
# env.py, _fixed_reward()
r = -STEP_COST
r += PICK_CREDIT if newly_picked else 0.0      # change 1: pay once per item
r += phi_after - phi_before                    # change 2: phi = -0.1 * distance
if goal:
    r += GOAL_BONUS
```

*Why this fix can't be tricked now*

- an unsolved shift earns at most `items − 0.1 + 0.1 × distance from desk to nearest item` (the worst case: picking every item but stopping one step before the desk)
- a solved shift earns at least `items + 5 − 0.01 × budget`

The good-shift line `items + 3` lies strictly between the two in every room.
In the tightest room (`bottleneck:2`), an unsolved shift earns at most 6.0, the
line is 8, and a solved one earns at least 9.41.


---

## Tests for the fix [`tests.py`](tests.py)

Run with `bash run.sh` or `python3 tests.py` 

| test | what it checks |
|---|---|
| [`test_naive_reward_is_exploitable`](tests.py#L143) | the old reward is fooled in 15 / 15 rooms |
| [`test_naive_reward_pays_for_closed_loops`](tests.py#L158) | pacing back and forth pays under the old reward |
| [`test_fix_blocks_the_exploit`](tests.py#L167) | the exact action lists of both loopholes, rescored with the fixed reward, stay below the good-shift line in every room (best: 1.21 points) |
| [`test_fixed_shaping_is_free_on_closed_loops`](tests.py#L176) | walking out and back earns exactly 0 |
| [`test_fixed_reward_has_no_false_positives_under_random_play`](tests.py#L201) | 45,000 random shifts: 4,903 fool the old reward, 0 fool the new one |
| [`test_fixed_reward_gap_is_proven_not_just_observed`](tests.py#L235) | the bounds from the fix hold in every room |
| [`test_fixed_reward_prefers_shorter_routes`](tests.py#L249) | a longer valid route still earns less than the perfect one |

The remaining tests cover the verifier and the environment.

---

## Possible next steps for a buyer

I assume the exploit, fix, and according tests have scaled up to many tasks.
The next thing a buyer would potentially ask about: the environment has only ever run on
one specific infrastructure. Before running it at a big scale on their
infrastructure, I would benchmark memory usage and runtime.

---