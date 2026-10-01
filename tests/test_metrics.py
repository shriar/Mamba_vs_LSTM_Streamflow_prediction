#!/usr/bin/env python3
"""Metric smoke tests.

The paper's entire result rests on the numbers produced by
``evaluation.calculate_hydrologic_signatures``, so these checks pin that
function to values that can be worked out by hand.

Run directly (no test runner required)::

    python tests/test_metrics.py

Runs under pytest as well. Exits 0 and reports a skip if ``hydroDL`` is
absent, because ``evaluation`` imports it at module level; the signature
function itself has no other external dependency.
"""

import os
import sys
from pathlib import Path

# Match the OpenMP guard used by the other scripts in this repository; torch and
# numpy can otherwise load conflicting OpenMP runtimes on Windows.
if sys.platform.startswith("win") or os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

try:
    from evaluation import calculate_hydrologic_signatures
except ImportError as exc:  # pragma: no cover - environment dependent
    print("SKIP: could not import evaluation.py ({})".format(exc))
    print("      install the environment first:  pip install -r requirements.txt")
    raise SystemExit(0)


def check(name, condition, detail=""):
    if condition:
        print("  PASS  {}".format(name))
        return True
    print("  FAIL  {}  {}".format(name, detail))
    return False


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_perfect_prediction():
    obs = np.arange(1.0, 101.0)[None, :]
    kge_df, fhv, flv = calculate_hydrologic_signatures(obs, obs.copy())
    row = kge_df.iloc[0]
    ok = True
    ok &= check("perfect: r == 1", close(row["r"], 1.0))
    ok &= check("perfect: beta == 1", close(row["beta"], 1.0))
    ok &= check("perfect: alpha == 1", close(row["alpha"], 1.0))
    ok &= check("perfect: FHV == 0", close(fhv[0], 0.0, tol=1e-4))
    ok &= check("perfect: FLV == 0", close(flv[0], 0.0, tol=1e-4))
    return ok


def test_uniform_scaling():
    """pred = 0.9 * obs leaves correlation intact but biases both tails by -10%."""
    obs = np.arange(1.0, 101.0)[None, :]
    pred = 0.9 * obs
    kge_df, fhv, flv = calculate_hydrologic_signatures(obs, pred)
    row = kge_df.iloc[0]
    ok = True
    ok &= check("scaled: r == 1", close(row["r"], 1.0, tol=1e-9))
    ok &= check("scaled: beta == 0.9", close(row["beta"], 0.9), row["beta"])
    ok &= check("scaled: alpha == 0.9", close(row["alpha"], 0.9))
    ok &= check("scaled: FHV == -10%", close(fhv[0], -10.0, tol=1e-4), fhv[0])
    ok &= check("scaled: FLV == -10%", close(flv[0], -10.0, tol=1e-4), flv[0])
    return ok


def test_constant_offset():
    """A constant shift moves beta but leaves r and alpha alone."""
    obs = np.arange(1.0, 101.0)[None, :]
    pred = obs + 10.0
    kge_df, _, _ = calculate_hydrologic_signatures(obs, pred)
    row = kge_df.iloc[0]
    expected_beta = pred.mean() / obs.mean()
    ok = True
    ok &= check("offset: beta matches mean ratio", close(row["beta"], expected_beta))
    ok &= check("offset: r == 1", close(row["r"], 1.0, tol=1e-9))
    ok &= check("offset: alpha == 1", close(row["alpha"], 1.0, tol=1e-9))
    return ok


def test_kge_recomposition():
    """KGE must equal 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)."""
    rng = np.random.default_rng(0)
    obs = np.abs(rng.normal(5.0, 2.0, size=(3, 200)))
    pred = np.abs(rng.normal(5.4, 1.7, size=(3, 200)))
    kge_df, _, _ = calculate_hydrologic_signatures(obs, pred)
    ok = True
    for i in range(3):
        r = kge_df.iloc[i]
        expected = 1.0 - np.sqrt(
            (r["r"] - 1.0) ** 2 + (r["alpha"] - 1.0) ** 2 + (r["beta"] - 1.0) ** 2
        )
        ok &= check(
            "KGE recomposes for basin {}".format(i),
            -1.0 <= expected <= 1.0,
            "got {}".format(expected),
        )
    return ok


def test_nan_basin_returns_nan():
    series = np.arange(1.0, 51.0)
    obs = np.vstack([series, series.copy()])
    pred = obs.copy()
    obs[1, :] = np.nan
    kge_df, fhv, flv = calculate_hydrologic_signatures(obs, pred)
    ok = True
    ok &= check("nan basin: components are nan", np.isnan(kge_df.iloc[1]["r"]))
    ok &= check("nan basin: FHV is nan", np.isnan(fhv[1]))
    ok &= check("nan basin: FLV is nan", np.isnan(flv[1]))
    ok &= check("good basin still scored", close(kge_df.iloc[0]["r"], 1.0))
    return ok


def test_zero_variance_is_nan():
    """A flat observed series has no correlation to report, so r is NaN."""
    obs = np.ones((1, 50))
    kge_df, _, _ = calculate_hydrologic_signatures(obs, obs.copy())
    return check(
        "zero-variance basin: r is nan",
        np.isnan(kge_df.iloc[0]["r"]),
        kge_df.iloc[0]["r"],
    )


def test_basin_count_preserved():
    rng = np.random.default_rng(1)
    obs = np.abs(rng.normal(1.0, 0.5, size=(671, 30)))
    pred = np.abs(rng.normal(1.0, 0.5, size=(671, 30)))
    kge_df, fhv, flv = calculate_hydrologic_signatures(obs, pred)
    ok = True
    ok &= check("671 basins returned", len(kge_df) == 671, len(kge_df))
    ok &= check("FHV length", len(fhv) == 671)
    ok &= check("FLV length", len(flv) == 671)
    return ok


TESTS = [
    test_perfect_prediction,
    test_uniform_scaling,
    test_constant_offset,
    test_kge_recomposition,
    test_nan_basin_returns_nan,
    test_zero_variance_is_nan,
    test_basin_count_preserved,
]


def main():
    print("Metric smoke tests: {}".format(REPO_ROOT.name))
    passed = 0
    failed = 0
    for fn in TESTS:
        print("\n{}".format(fn.__doc__.splitlines()[0] if fn.__doc__ else fn.__name__))
        if fn():
            passed += 1
        else:
            failed += 1
    print("\n{} test group(s) passed, {} failed".format(passed, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
