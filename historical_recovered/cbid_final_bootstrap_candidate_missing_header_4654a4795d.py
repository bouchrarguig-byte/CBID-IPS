from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

root = Path.home() / "Downloads"
audit = root / "CBID_FINAL_AUDIT"

out = audit / "11_final_bootstrap"
out.mkdir(parents=True, exist_ok=True)

protocol_path = (
    audit / "09_final_protocol/final_protocol.json"
)

expected = (
    "28ddf8fb93986db0fa99ee5f79378e3c4"
    "ec777b8358c6e9033f78a47ea0b8e2e"
)

assert hashlib.sha256(
    protocol_path.read_bytes()
).hexdigest() == expected

protocol = json.loads(protocol_path.read_text())

comparisons = protocol["primary_comparisons"]
seeds = protocol["seeds"]

bp = protocol["bootstrap"]

N_BOOT = bp["replicates"]

minutes_list = [
    bp["primary_block_minutes"],
    *bp["sensitivity_block_minutes"]
]

alpha = (
    bp["familywise_alpha"]
    / bp["primary_comparison_count"]
)

models = sorted({
    m for pair in comparisons for m in pair
})

keys = ["src", "dst", "w"]

rows = []
replicates = []

def prepare_ap(y, score, blocks):

    order = np.argsort(
        -score, kind="stable"
    )

    yy = y[order].astype(np.float64)
    ss = score[order]
    bb = blocks[order]

    ends = np.flatnonzero(
        np.r_[ss[1:] != ss[:-1], True]
    )

    def weighted_ap(counts):

        weights = counts[bb]

        tp = np.cumsum(weights * yy)[ends]
        total = np.cumsum(weights)[ends]

        if tp[-1] == 0:
            return np.nan

        precision = np.divide(
            tp, total,
            out=np.zeros_like(tp),
            where=total > 0
        )

        increments = np.diff(
            np.r_[0.0, tp]
        )

        return float(
            np.sum(precision * increments)
            / tp[-1]
        )

    return weighted_ap

for day in ["E", "F"]:

    path = (
        audit / "10_final_confirmation"
        / day / "inference/predictions.csv"
    )

    d = pd.read_csv(
        path,
        usecols=keys + [
            "seed", "model", "score", "y"
        ]
    )

    d = d[d.model.isin(models)]

    assert not d.duplicated(
        keys + ["seed", "model"]
    ).any()

    assert set(d.model.unique()) == set(models)
    assert np.isfinite(d.score).all()
    assert set(d.y.unique()) <= {0, 1}

    ref = d[
        (d.seed == seeds[0]) &
        (d.model == models[0])
    ].sort_values(keys)

    idx = pd.MultiIndex.from_frame(ref[keys])

    y = ref.y.to_numpy()
    w = ref.w.to_numpy()

    aligned = {}

    for seed in seeds:
        for model in models:

            g = d[
                (d.seed == seed) &
                (d.model == model)
            ].set_index(keys)

            assert len(g) == len(idx)

            g = g.reindex(idx)

            assert g.score.notna().all()
            assert np.array_equal(g.y.to_numpy(), y)

            aligned[seed, model] = g.score.to_numpy()

    del d

    if len(np.unique(y)) < 2:

        for minutes in minutes_list:
            for a, b in comparisons:
                rows.append({
                    "day": day,
                    "block_minutes": minutes,
                    "comparison": f"{a} - {b}",
                    "verdict": "INCONCLUSIVE_SINGLE_CLASS"
                })

        continue

    original = {
        k: average_precision_score(y, v)
        for k, v in aligned.items()
    }

    observed = np.array([
        np.mean([
            original[seed, a] - original[seed, b]
            for seed in seeds
        ])
        for a, b in comparisons
    ])

    for minutes in minutes_list:

        _, blocks = np.unique(
            w // (minutes * 6),
            return_inverse=True
        )

        n = int(blocks.max()) + 1

        if n < 2:

            for j, (a, b) in enumerate(comparisons):
                rows.append({
                    "day": day,
                    "block_minutes": minutes,
                    "blocks": n,
                    "comparison": f"{a} - {b}",
                    "observed_delta_ap": observed[j],
                    "verdict": "INCONCLUSIVE_BLOCK_SUPPORT"
                })

            continue

        pos = np.bincount(
            blocks, weights=y, minlength=n
        )

        neg = np.bincount(
            blocks, weights=1-y, minlength=n
        )

        funcs = {
            k: prepare_ap(y, v, blocks)
            for k, v in aligned.items()
        }

        ones = np.ones(n, dtype=np.int64)

        for k, fn in funcs.items():
            assert np.isclose(
                fn(ones), original[k],
                atol=1e-10
            )

        rng = np.random.default_rng(
            bp["random_seed"]
        )

        samples = []

        for i in range(N_BOOT):

            counts = np.bincount(
                rng.integers(0, n, size=n),
                minlength=n
            )

            if counts @ pos == 0:
                continue

            if counts @ neg == 0:
                continue

            ap = {
                k: fn(counts)
                for k, fn in funcs.items()
            }

            delta = [
                np.mean([
                    ap[seed, a] - ap[seed, b]
                    for seed in seeds
                ])
                for a, b in comparisons
            ]

            if not np.isfinite(delta).all():
                continue

            samples.append(delta)

            for j, (a, b) in enumerate(comparisons):

                replicates.append({
                    "day": day,
                    "block_minutes": minutes,
                    "replicate": i,
                    "comparison": f"{a} - {b}",
                    "delta_ap": delta[j]
                })

            if (i + 1) % 1000 == 0:

                print(
                    day, minutes,
                    "min:", i+1, "/", N_BOOT,
                    flush=True
                )

        samples = np.asarray(samples)

        primary = (
            minutes == bp["primary_block_minutes"]
        )

        enough_support = (
            n >= bp["minimum_primary_blocks"]
        )

        enough_reps = (
            len(samples)
            >= bp["minimum_valid_replicates"]
        )

        for j, (a, b) in enumerate(comparisons):

            if len(samples) == 0:
                lo = hi = np.nan
            else:
                lo, hi = np.quantile(
                    samples[:, j],
                    [alpha/2, 1-alpha/2]
                )

            if not enough_reps:
                verdict = "INCONCLUSIVE_REPLICATES"

            elif primary and not enough_support:
                verdict = "INCONCLUSIVE_BLOCK_SUPPORT"

            elif lo > 0:
                verdict = "POSITIVE"

            elif hi < 0:
                verdict = "NEGATIVE"

            else:
                verdict = "INCONCLUSIVE"

            rows.append({
                "day": day,
                "block_minutes": minutes,
                "blocks": n,
                "valid_bootstraps": len(samples),
                "comparison": f"{a} - {b}",
                "observed_delta_ap": observed[j],
                "ci_lower": lo,
                "ci_upper": hi,
                "confidence_level": 1-alpha,
                "primary_analysis": primary,
                "verdict": verdict
            })

        print(
            "Completed:",
            day, minutes,
            "minutes;",
            n, "blocks;",
            len(samples), "replicates",
            flush=True
        )

r = pd.DataFrame(rows)

r.to_csv(
    out / "final_bootstrap_summary.csv",
    index=False
)

pd.DataFrame(replicates).to_csv(
    out / "final_bootstrap_replicates.csv",
    index=False
)

(out / "bootstrap_metadata.json").write_text(
    json.dumps({
        "protocol_sha256": expected,
        "replicates": N_BOOT,
        "confidence_level": 1-alpha,
        "correction": "Bonferroni",
        "retraining": False,
        "recalibration": False
    }, indent=2)
)

print("\n=== PRIMARY CONFIRMATORY RESULTS ===")

print(
    r[r.block_minutes == 15]
    .round(6).to_string(index=False)
)

print("\n=== SENSITIVITY ANALYSIS ===")

print(r.round(6).to_string(index=False))

print("\nSaved:", out)
