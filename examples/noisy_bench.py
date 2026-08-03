"""A seeded stochastic "benchmark" for the subprocess-condition demo (M10).

Stands in for any program whose quality you want to measure statistically: it
takes a seed and prints a score in [1, 10] on the final stdout line. Same
arguments + same seed => byte-identical output (safe to cache); different seeds
=> different noise draws. The two algorithms differ in true mean quality, so
`mimir analyze` has a real effect to detect.

Runs on any python3 (stdlib only); mimir itself never imports this file.
"""

import argparse
import random

BASE_SCORE = {"fast": 8.2, "slow": 6.8}


def main() -> None:
    parser = argparse.ArgumentParser(description="seeded noisy benchmark")
    parser.add_argument("--algo", choices=sorted(BASE_SCORE), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--difficulty", type=float, required=True)
    args = parser.parse_args()

    # Seed on (seed, algo, difficulty) so the noise is deterministic per run but
    # NOT identical across algorithms — identical noise would cancel in every
    # paired diff and make the comparison degenerate.
    rng = random.Random(f"{args.seed}:{args.algo}:{args.difficulty}")
    score = BASE_SCORE[args.algo] - 0.8 * args.difficulty + rng.gauss(0.0, 0.7)
    score = max(1.0, min(10.0, score))

    print(f"algo={args.algo} difficulty={args.difficulty} seed={args.seed}")
    print(f"{score:.4f}")


if __name__ == "__main__":
    main()
