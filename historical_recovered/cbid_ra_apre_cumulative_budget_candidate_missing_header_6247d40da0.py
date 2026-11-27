import pandas as pd
import numpy as np
from pathlib import Path

root = Path.home() / "Downloads"

out = root / "CBID_RA_APRE_V2/05_cumulative_budget"
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

        groups = list(
            g.groupby("w", sort=True).indices.values()
        )

        for q in [
            .005, .01, .02, .03, .05, .10
        ]:

            for policy in ["floor", "cumulative"]:

                trigger = np.zeros(
                    len(g), dtype=bool
                )

                credit = 0.0
                seen = 0
                actions = 0

                for positions in groups:

                    n = len(positions)
                    seen += n

                    if policy == "floor":
                        k = int(np.floor(q*n))

                    else:
                        credit = min(
                            10.0,
                            credit + q*n
                        )

                        k = min(
                            n,
                            int(np.floor(credit))
                        )

                        credit -= k

                    trigger[positions[:k]] = True
                    actions += k

                    assert actions <= q*seen + 1e-7

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
                    "policy": policy,
                    "action_budget": q,
                    "trigger_rate": trigger.mean(),
                    "triggers": int(trigger.sum()),
                    "attack_exposure_rate": ea,
                    "benign_exposure_rate": eb,
                    "benign_budget_pass": eb <= .01
                })

r = pd.DataFrame(rows)

r.to_csv(
    out / "cumulative_metrics.csv",
    index=False
)

s = r.groupby([
    "day", "model", "policy", "action_budget"
])[[
    "trigger_rate",
    "attack_exposure_rate",
    "benign_exposure_rate",
    "benign_budget_pass"
]].mean()

s.to_csv(
    out / "cumulative_summary.csv"
)

print("\n=== TCAG ===")

print(
    s.xs(
        "G1_real_peer", level="model"
    ).round(6).to_string()
)

print("\n=== XGBOOST ===")

print(
    s.xs(
        "M3_full", level="model"
    ).round(6).to_string()
)

print("\n=== ALL MODELS: DAY D ===")

print(
    s.xs(
        "D", level="day"
    ).round(6).to_string()
)

print("\n=== ROBUST ACTIVE POLICIES ===")

print(
    s[
        (s.benign_budget_pass == 1) &
        (s.attack_exposure_rate > 0)
    ].round(6).to_string()
)

print("\nSaved:", out)
