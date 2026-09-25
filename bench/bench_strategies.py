"""Compare the fine-grained (master-worker) and hybrid (island-parallel)
strategies on a PMLB dataset, reusing the existing ``bench.eval`` driver.

This exercises the *expensive, load-imbalanced* fitness regime: the PMLB config uses
``use_constants: True``, so each individual with constants triggers a pygmo PSO run.

Run from the ``bench`` directory:

    python bench_strategies.py 192_vineyard
"""

import argparse
import time

import ray

from bench import eval as bench_eval


def warmup_workers():
    """Force every Ray worker to start and import the heavy dependencies.

    The first batch of fitness tasks otherwise pays a large one-time cost: ~10 fresh
    worker processes import pygmo/mygrad/flex and run their first pygmo solve. Whichever
    strategy is timed first absorbs this cold-start, which would inflate its time and
    the reported speed-up. Warming the workers up front makes both strategies timed in
    the same (warm) steady state.
    """
    n = max(int(ray.available_resources().get("CPU", 1)), 1)

    @ray.remote
    def _warm():
        import time as _t
        import bench  # noqa: F401  (imports pygmo, mygrad, flex, ...)
        import flex.gp.regressor  # noqa: F401  (registers creator classes)

        _t.sleep(0.3)  # hold the slot so the batch spreads across all workers
        return True

    t0 = time.perf_counter()
    ray.get([_warm.remote() for _ in range(n * 2)])
    print(f"[warmup] {n} workers warmed in {time.perf_counter() - t0:.2f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", nargs="?", default="192_vineyard",
                        help="Name of a (cached) PMLB dataset.")
    parser.add_argument("--cfg", default="PMLB_quick.yaml",
                        help="Path of the YAML config file.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)

    print(f"\n### PROBLEM: {args.problem}  (cfg={args.cfg}, seed={args.seed})\n")

    # warm the workers so neither strategy is charged the one-time cold-start
    warmup_workers()

    print("=== FINE-GRAINED (original master-worker) ===")
    t0 = time.perf_counter()
    r2tr_f, r2te_f, model_f, fit_f = bench_eval(
        problem=args.problem, cfgfile=args.cfg, seed=args.seed,
        coarse_grained_islands=False, remove_init_duplicates=False,
    )
    wall_f = time.perf_counter() - t0

    print("\n=== HYBRID (island coordinators + cluster-wide fitness) ===")
    t0 = time.perf_counter()
    r2tr_h, r2te_h, model_h, fit_h = bench_eval(
        problem=args.problem, cfgfile=args.cfg, seed=args.seed,
        coarse_grained_islands="hybrid", remove_init_duplicates=False,
    )
    wall_h = time.perf_counter() - t0

    print("\n" + "=" * 74)
    print(f"{'Strategy':<18}{'fit time (s)':>14}{'wall (s)':>12}"
          f"{'R2 train':>12}{'R2 test':>12}")
    print("-" * 74)
    print(f"{'fine-grained':<18}{fit_f:>14.2f}{wall_f:>12.2f}"
          f"{r2tr_f:>12.3f}{r2te_f:>12.3f}")
    print(f"{'hybrid':<18}{fit_h:>14.2f}{wall_h:>12.2f}"
          f"{r2tr_h:>12.3f}{r2te_h:>12.3f}")
    print("-" * 74)
    speedup_h = fit_f / fit_h if fit_h > 0 else float("nan")
    print(f"Speed-up on fit time (fine / hybrid): {speedup_h:.2f}x")
    print("=" * 74)
    print(f"best (fine)   = {model_f}")
    print(f"best (hybrid) = {model_h}")

    ray.shutdown()


if __name__ == "__main__":
    main()
