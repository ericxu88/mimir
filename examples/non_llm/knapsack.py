"""Seeded knapsack-optimizer benchmark — the non-LLM validation target (M11).

Two REAL randomized algorithms compete on 0/1-knapsack instances under the same
evaluation budget:

- restart: random-restart hill climbing (bit-flip neighborhood, restart on stall)
- anneal:  simulated annealing (bit-flip proposals, geometric cooling)

Each run prints a quality score on the final stdout line: 1 + 9 * best/optimum,
in [1, 10] (the exact optimum comes from an in-process dynamic program). Same
--instance/--seed => byte-identical output (safe for mimir's content-addressed
cache); different seeds => different search trajectories. Both algorithms
consume the SAME seeded stream per (instance, seed) — common random numbers.

Stdlib only; runs on any python3. mimir itself never imports this file.
"""

import argparse
import math
import random

INSTANCES = ("alfa", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel")
N_ITEMS = 15
EVAL_BUDGET = 500
ANNEAL_T0 = 100.0
ANNEAL_T_END = 1.0


def make_instance(name):
    """Deterministic instance: values/weights/capacity derived from the name."""
    rng = random.Random("instance:" + name)
    values = [rng.randint(10, 100) for _ in range(N_ITEMS)]
    weights = [rng.randint(5, 30) for _ in range(N_ITEMS)]
    capacity = int(sum(weights) * 0.4)
    return values, weights, capacity


def optimum(values, weights, capacity):
    """Exact 0/1-knapsack optimum via dynamic programming."""
    # Index loops, no zip(): this file must run on any bare python3 (3.9+),
    # where zip(strict=...) does not exist yet.
    best = [0] * (capacity + 1)
    for index, value in enumerate(values):
        weight = weights[index]
        for c in range(capacity, weight - 1, -1):
            if best[c - weight] + value > best[c]:
                best[c] = best[c - weight] + value
    return best[capacity]


def pack_value(mask, values, weights, capacity):
    """Total value of a selection, or None when it exceeds capacity."""
    weight = sum(weights[i] for i, bit in enumerate(mask) if bit)
    if weight > capacity:
        return None
    return sum(values[i] for i, bit in enumerate(mask) if bit)


def restart_search(rng, values, weights, capacity, budget):
    """Hill climbing with random restarts; every candidate costs one evaluation."""
    n = len(values)
    best = 0
    evals = 0
    while evals < budget:
        mask = [rng.random() < 0.3 for _ in range(n)]
        current = pack_value(mask, values, weights, capacity)
        evals += 1
        if current is None:
            current = 0
            mask = [False] * n
        stall = 0
        while evals < budget and stall < 20:
            i = rng.randrange(n)
            mask[i] = not mask[i]
            candidate = pack_value(mask, values, weights, capacity)
            evals += 1
            if candidate is not None and candidate > current:
                current = candidate
                stall = 0
            else:
                mask[i] = not mask[i]
                stall += 1
        if current > best:
            best = current
    return best


def anneal_search(rng, values, weights, capacity, budget):
    """Simulated annealing: accept worse moves with prob exp(delta/T), T cooling
    geometrically from ANNEAL_T0 to ANNEAL_T_END over the budget."""
    n = len(values)
    mask = [False] * n
    current = 0
    best = 0
    temp = ANNEAL_T0
    cooling = (ANNEAL_T_END / ANNEAL_T0) ** (1.0 / budget)
    for _ in range(budget):
        i = rng.randrange(n)
        mask[i] = not mask[i]
        candidate = pack_value(mask, values, weights, capacity)
        if candidate is None:
            mask[i] = not mask[i]
        else:
            delta = candidate - current
            if delta >= 0 or rng.random() < math.exp(delta / temp):
                current = candidate
                if current > best:
                    best = current
            else:
                mask[i] = not mask[i]
        temp *= cooling
    return best


ALGORITHMS = {"restart": restart_search, "anneal": anneal_search}


def main():
    parser = argparse.ArgumentParser(description="seeded knapsack optimizer benchmark")
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), required=True)
    parser.add_argument("--instance", choices=INSTANCES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    values, weights, capacity = make_instance(args.instance)
    target = optimum(values, weights, capacity)
    # Seed on (seed, instance) but NOT the algorithm: both arms face the same
    # random stream per unit — common random numbers, mirroring mimir's CRN seeds.
    rng = random.Random(f"{args.seed}:{args.instance}")
    best = ALGORITHMS[args.algo](rng, values, weights, capacity, EVAL_BUDGET)

    score = 1.0 + 9.0 * (best / target)
    print(
        f"instance={args.instance} algo={args.algo} seed={args.seed} best={best} optimum={target}"
    )
    print(f"{score:.4f}")


if __name__ == "__main__":
    main()
