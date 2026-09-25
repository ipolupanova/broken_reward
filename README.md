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


## The setting [`Variant`](variants.py#L39)

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

## The task [`env.py`](env.py), [`OrderPickingEnv`](env.py#L63)

The picker has to pick every item ([`Variant.items`](variants.py#L47)) of the order and bring them back to the
desk ([`Variant.depot`](variants.py#L46)) before the shift ends. The shift is a fixed budget of moves ([`Variant.budget`](variants.py#L48)).

- [`reset()`](env.py#L74): the picker clocks in,([`_State.picked`](env.py#L56))
- [`step(action)`](env.py#L80): the picker does one action, which costs one move ([`_State.steps`](env.py#L57))
- [`observation()`](env.py#L198): everything the picker knows at this moment, for example could look like: 

```python
obs = {
    "position":           (0, 0),            # where the picker is standing
    "remaining":          (1, 1, 1, 1, 1),   # 1 = item still on the shelf
    "steps_used":         0,
    "steps_left":         41,
    "distance_to_target": 5,                 # walking distance to the nearest remaining item
    # plus the room map, the desk position and the item positionsblabla
}
```

The picker can do exactly five things ([action ids](variants.py#L25)):

| action | id | what happens |
|---|---|---|
| step north | `0` | one cell up ([`MOVES`](variants.py#L26)) |
| step south | `1` | one cell down |
| step west | `2` | one cell left |
| step east | `3` | one cell right |
| pick | `4` | put the item he is standing at into the cart ([`env.py`](env.py#L127-L131)) |


### The reward [`_naive_reward()`](env.py#L162)

Management sets up a reward system ([`_naive_reward()`](env.py#L162)) to keep the picker motivated:

- **1 point**  for every pick on an item's cell
- **0.1 points**  for every step that brings the picker closer to the nearest item still on the order ([`_distance_to_target()`](env.py#L141))
- **5 points** for bringing the complete order back to the desk
- **−0.01 points** for every move, so dawdling costs something

Management looks only at the points, and counts a shift as a *good shift* if it
earns at least **number of items + 3** ([`success_threshold()`](env.py#L48)) (8 points for an order of five-items). 

## Three task variants = three kinds of room [`variants.py`](variants.py), [`make_variant()`](variants.py#L243)

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

## Baseline agents [`baseline.py`](baseline.py), [`evaluate()`](baseline.py#L75)

| picker | strategy | solved |
|---|---|---|
| [`exact_agent`](baseline.py#L43) | follows the perfect route | 15 / 15 |
| [`nearest_neighbour_agent`](baseline.py#L28) | always walks to the closest remaining item | 14 / 15 (runs out of moves in `aisle_racks:1`) |
| [`random_agent`](baseline.py#L58) | random actions | 0 / 15 (always makes an illegal move early) |

[`evaluate()`](baseline.py#L75) runs a picker on all rooms and asks the
verifier. This shows every room is solvable.

```
--- nearest_neighbour: success rate 93% (14/15)
--- exact_planner: success rate 100% (15/15)
--- random: success rate 0% (0/15)
```

`python3 baseline.py` prints every room for every agent and the success rates

---

### The verifier [`verifier.py`](verifier.py), [`verify()`](verifier.py#L24)

The verifier checks
whether the task was actually solved: `verify(final_state)` returns 1 or 0. 

---

## The exploit [`exploit.py`](exploit.py)

A sneaky picker reads the reward system and finds two loopholes:

1. **Pick spam**: every pick on an item's cell pays 1 point, even if that item has already been picked. So the picker walks to the nearest item and repeats "pick" until the move budget runs out.
2. **Pacing backand forth**: a step closer to an item pays 0.1, a step away costs nothing. Stepping back and forth earns points without completing the task.


```python
# env.py, _naive_reward()
if action == PICK and item_idx is not None:
    r += PICK_CREDIT          # paid for every pick, even of an item already in the cart
if d_after < d_before:
    r += SHAPING_SCALE        # paid for getting closer, never charged for walking away
```

---

## The fix [`_fixed_reward()`](env.py#L180)

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

*Why this fix can't be tricked now * 

- an unsolved shift earns at most `(items − 1) + 0.1 × distance from desk to nearest item`
- a solved shift earns at least `items + 5 − 0.01 × budget`

The good-shift line `items + 3` lies strictly between the two in every room.
In the tightest room (`bottleneck:2`), an unsolved shift earns at most 5.1, the
line is 8, and a solved one earns at least 9.41.


---

## Tests for the fix [`tests.py`](tests.py)

Run with `bash run.sh` or `python3 tests.py` 

| test | what it checks |
|---|---|
| [`test_naive_reward_is_exploitable`](tests.py#L170) | the old reward is fooled in 15 / 15 rooms |
| [`test_naive_reward_pays_for_closed_loops`](tests.py#L186) | pacing back and forth pays under the old reward |
| [`test_fix_blocks_the_exploit`](tests.py#L197) | the exact action lists of both loopholes, rescored with the fixed reward, stay below the good-shift line in every room (best: 1.21 points) |
| [`test_fixed_shaping_is_free_on_closed_loops`](tests.py#L207) | walking out and back earns exactly 0 |
| [`test_fixed_reward_has_no_false_positives_under_random_play`](tests.py#L233) | 45,000 random shifts: 4,903 fool the old reward, 0 fool the new one |
| [`test_fixed_reward_gap_is_proven_not_just_observed`](tests.py#L268) | the bounds from the fix hold in every room |
| [`test_fixed_reward_prefers_shorter_routes`](tests.py#L281) | a longer valid route still earns less than the perfect one |

The remaining tests cover the verifier and the environment.

---

## Possible next steps for a buyer

I assume the exploit, fix, and according tests have scaled up to many tasks.
The next thing a buyer would potentially ask about: the environment has only ever run on
one specific infrastructure. Before running it at a big scale on their
infrastructure, I would take a look at memory usage and runtime (through a benchmark) and use multiple optimization strategies. 

---

