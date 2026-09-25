"""One command that runs everything: variants, baselines, exploit, tests.

    python3 run_all.py
"""

from __future__ import annotations

import time

import baseline
import exploit
import test_suite
from variants import all_variants, make_variant


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78 + "\n")


def main() -> int:
    section("1. TASK VARIANTS (generated, deterministic)")
    variants = all_variants()
    print(f"{'variant':<16}{'items':>6}{'optimal':>9}{'budget':>8}   ('.' floor  '#' rack  'D' depot  'x' item)")
    for v in variants:
        print(f"{v.vid:<16}{v.n_items:>6}{v.optimal_actions:>9}{v.budget:>8}")
    for family in ("open_floor", "aisle_racks", "bottleneck"):
        demo = make_variant(family, 0)
        print(f"\n{demo.vid}:")
        print(demo.render(pos=demo.depot))

    section("2. BASELINE AGENTS (no training, success judged by the verifier)")
    baseline.report()

    section("3. REWARD EXPLOIT AND FIX")
    exploit.report()

    section("4. TESTS")
    start = time.time()
    rc = test_suite.main()
    print(f"({time.time() - start:.0f}s)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
