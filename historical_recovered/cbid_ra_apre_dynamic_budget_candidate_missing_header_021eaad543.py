import pandas as pd
import numpy as np
from pathlib import Path

root = Path.home() / "Downloads"

out = root / "CBID_RA_APRE_V2/04_dynamic_budget"
out.mkdir(parents=True, exist_ok=True)

paths = {
    "C": root / "CBID_TCAG_V10/05_confirmation/predictions.csv",
    "D": root / "CBID_APRE_V1/05_confirmation/inference/predictions.csv"
}

models = [
    "M0_local",
    "M1_temporal",
    "M2_context",
    "M3_full",
    "G0_no_message",
    "G1_real_peer",
    "G2_shuffled_peer",
    "G3_context_mlp",
    "H1_fixed"
]

rows = []

for day, path in paths.items():

    d = pd.read_csv(
        path,
        usecols=[
            "seed", "model", "src", "dst",
            "w", "score", "flows", "attack_flows"
        ]
    )

    d = d[d.model.isin(models)]

    assert not d.duplicated(
        ["seed", "model", "src", "dst", "w"]
    ).any()

    assert np.isfinite(d.score).all()

    for (model, seed), g in d.groupby(
        ["model", "seed"]
    ):

        g = g.sort_values(
            ["w", "score", "src", "dst"],
            ascending=[True, False, True, True]
        ).reset_index(drop=True)

        a = g.attack_flows.to_numpy()
        b = g.flows.to_numpy() - a

        idx = pd.MultiIndex.from_frame(
            g[["src", "dst", "w"]]
        )

        prev_idx = pd.MultiIndex.from_arrays([
            g.src,
            g.dst,
            g.w - 1
        ], names=["src", "dst", "w"])

        n = g.groupby("w")["score"].transform(
            "size"
        ).to_numpy()

        rank = g.groupby("w").cumcount().to_numpy()

        for budget in [
            .005, .01, .02, .05, .10, .20, .30
        ]:

            k = np.floor(budget*n).astype(int)

            trigger = rank < k

            exposed = pd.Series(
                trigger, index=idx
            ).reindex(
                prev_idx, fill_value=False
            ).to_numpy(dtype=bool)

            ea = a[exposed].sum()/a.sum()
            eb = b[exposed].sum()/b.sum()

            rows.append({
                "day": day,
                "seed": seed,
                "model": model,
                "action_budget": budget,
                "trigger_rate": trigger.mean(),
                "attack_exposure_rate": ea,
                "benign_exposure_rate": eb,
                "benign_budget_pass": eb <= .01
            })

r = pd.DataFrame(rows)

r.to_csv(
    out / "dynamic_budget_metrics.csv",
    index=False
)

summary = r.groupby([
    "day", "model", "action_budget"
])[[
    "trigger_rate",
    "attack_exposure_rate",
    "benign_exposure_rate",
    "benign_budget_pass"
]].mean()

summary.to_csv(
    out / "dynamic_budget_summary.csv"
)

print("\n=== DYNAMIC BUDGET: TCAG ===")

print(
    summary.xs(
        "G1_real_peer", level="model"
    ).round(6).to_string()
)

print("\n=== DYNAMIC BUDGET: XGBOOST ===")

print(
    summary.xs(
        "M3_full", level="model"
    ).round(6).to_string()
)

print("\n=== ALL MODELS: DAY D ===")

print(
    summary.xs(
        "D", level="day"
    ).round(6).to_string()
)

print("\n=== ROBUST CONFIGURATIONS ===")

print(
    summary[
        summary.benign_budget_pass == 1
    ].round(6).to_string()
)

print("\nSaved:", out)
