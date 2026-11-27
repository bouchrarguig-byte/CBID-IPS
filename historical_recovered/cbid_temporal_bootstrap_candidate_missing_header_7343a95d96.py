from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path.home() / "Downloads"

out = root / "CBID_FINAL_AUDIT/07_temporal_bootstrap"
out.mkdir(parents=True, exist_ok=True)

paths = {
    "C": root / "CBID_TCAG_V10/05_confirmation/predictions.csv",
    "D": root / "CBID_APRE_V1/05_confirmation/inference/predictions.csv"
}

comparisons = [
    ("G1_real_peer", "G0_no_message"),
    ("G1_real_peer", "G2_shuffled_peer"),
    ("G1_real_peer", "G3_context_mlp"),
    ("G1_real_peer", "M3_full"),
    ("H1_fixed", "M3_full"),
    ("H1_fixed", "H2_fixed")
]

models = sorted({
    m for pair in comparisons for m in pair
})

seeds = [42, 43, 44]

N_BOOT = 1000

BLOCK_MINUTES = [5, 15, 60]

keys = ["src", "dst", "w"]

rows = []
replicates = []

def prepare_ap(y, score, blocks):

    order = np.argsort(
        -score, kind="stable"
    )

    y = y[order].astype(np.float64)
    score = score[order]
    blocks = blocks[order]

    ends = np.flatnonzero(
        np.r_[score[1:] != score[:-1], True]
    )

    def weighted_ap(counts):

        weights = counts[blocks]

        positives = weights * y

        tp = np.cumsum(positives)[ends]
        total = np.cumsum(weights)[ends]

        if tp[-1] == 0:
            return np.nan

        precision = np.divide(
            tp,
            total,
            out=np.zeros_like(tp),
            where=total > 0
        )

        increments = np.diff(
            np.r_[0.0, tp]
        )

        return float(
            np.sum(
                precision * increments
            ) / tp[-1]
        )

    return weighted_ap

for day, path in paths.items():

    d = pd.read_csv(
        path,
        usecols=keys + [
            "seed", "model", "score", "y"
        ]
    )

    print(
        "\nAvailable models:",
        sorted(d.model.unique()),
        flush=True
    )

    missing = set(models) - set(d.model)

    assert not missing, (
        f"Missing models: {missing}"
    )

    d = d[d.model.isin(models)]

    assert not d.duplicated(
        keys + ["seed", "model"]
    ).any()

    assert np.isfinite(d.score).all()

    assert set(d.y.unique()) <= {0, 1}

    reference = (
        d[
            (d.seed == seeds[0]) &
            (d.model == models[0])
        ]
        .sort_values(keys)
        .reset_index(drop=True)
    )

    idx = pd.MultiIndex.from_frame(
        reference[keys]
    )

    y = reference.y.to_numpy()

    w = reference.w.to_numpy()

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

            assert np.array_equal(
                g.y.to_numpy(), y
            )

            aligned[seed, model] = (
                g.score.to_numpy()
            )

    del d

    for minutes in BLOCK_MINUTES:

        raw_block = w // (minutes * 6)

        unique_blocks, blocks = np.unique(
            raw_block, return_inverse=True
        )

        n_blocks = len(unique_blocks)

        assert n_blocks >= 2

        funcs = {
            key: prepare_ap(y, score, blocks)
            for key, score in aligned.items()
        }

        ones = np.ones(
            n_blocks, dtype=np.int64
        )

        original = {
            key: fn(ones)
            for key, fn in funcs.items()
        }

        observed = np.array([
            np.mean([
                original[seed, a]
                - original[seed, b]
                for seed in seeds
            ])
            for a, b in comparisons
        ])

        rng = np.random.default_rng(
            2026 + minutes
            + (0 if day == "C" else 100)
        )

        samples = []

        for i in range(N_BOOT):

            chosen = rng.integers(
                0, n_blocks, size=n_blocks
            )

            counts = np.bincount(
                chosen,
                minlength=n_blocks
            )

            if np.dot(counts[blocks], y) == 0:
                continue

            if np.dot(
                counts[blocks], 1-y
            ) == 0:
                continue

            ap = {
                key: fn(counts)
                for key, fn in funcs.items()
            }

            delta = np.array([
                np.mean([
                    ap[seed, a]
                    - ap[seed, b]
                    for seed in seeds
                ])
                for a, b in comparisons
            ])

            if np.isfinite(delta).all():
                samples.append(delta)

                for j, (a, b) in enumerate(
                    comparisons
                ):
                    replicates.append({
                        "day": day,
                        "block_minutes": minutes,
                        "replicate": i,
                        "comparison": f"{a} - {b}",
                        "delta_ap": delta[j]
                    })

        samples = np.asarray(samples)

        assert len(samples) > 0

        for j, (a, b) in enumerate(
            comparisons
        ):

            v = samples[:, j]

            lo, hi = np.quantile(
                v, [.025, .975]
            )

            rows.append({
                "day": day,
                "block_minutes": minutes,
                "blocks": n_blocks,
                "valid_bootstraps": len(v),
                "comparison": f"{a} - {b}",
                "observed_delta_ap": observed[j],
                "bootstrap_mean": v.mean(),
                "bootstrap_sd": v.std(ddof=1),
                "ci95_lower": lo,
                "ci95_upper": hi,
                "fraction_positive": (v > 0).mean(),
                "ci_excludes_zero":
                    bool(lo > 0 or hi < 0)
            })

        print(
            "Completed:",
            day,
            minutes,
            "minutes;",
            n_blocks,
            "blocks;",
            len(samples),
            "replicates",
            flush=True
        )

r = pd.DataFrame(rows)

r.to_csv(
    out / "bootstrap_summary.csv",
    index=False
)

pd.DataFrame(replicates).to_csv(
    out / "bootstrap_replicates.csv",
    index=False
)

protocol = {
    "metric": "Average Precision",
    "block_minutes": BLOCK_MINUTES,
    "bootstrap_replicates": N_BOOT,
    "seeds": seeds,
    "paired_blocks": True,
    "seed_resampling": False,
    "confidence_interval": "percentile_95",
    "days": ["C", "D"],
    "exploratory_analysis": True,
    "new_training": False,
    "reserved_days_accessed": False
}

(out / "bootstrap_protocol.json").write_text(
    json.dumps(protocol, indent=2)
)

print("\n=== PRIMARY BLOCK SIZE: 15 MIN ===")

print(
    r[r.block_minutes == 15]
    .round(6)
    .to_string(index=False)
)

print("\n=== BLOCK SIZE SENSITIVITY ===")

print(
    r[[
        "day",
        "block_minutes",
        "comparison",
        "observed_delta_ap",
        "ci95_lower",
        "ci95_upper",
        "ci_excludes_zero"
    ]].round(6).to_string(index=False)
)

print("\nSaved:", out)
