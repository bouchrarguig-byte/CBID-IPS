import sys
import json
import time
import hashlib
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

root = Path.home() / "Downloads"
sys.path.insert(0, str(root))

from cbid_tcag_v10_04_confirmation import specs
from cbid_tcag_v9_06_graph_prototype import Model, pack

out = root / "CBID_FINAL_AUDIT/06_computational_cost"
out.mkdir(parents=True, exist_ok=True)

frozen = root / "CBID_TCAG_V10/04_frozen_models"

data_path = (
    root / "CBID_TCAG_V10/05_confirmation/features"
    / "pair_windows_2019-04-26.csv"
)

def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

manifest = json.loads(
    (frozen / "freeze.json").read_text()
)

assert sha(frozen / "freeze.json") == (
    "92198e78d35047750afc34edf96c9f482"
    "a81a746c302b7170fc8a857cbc2952e"
)

for name, digest in manifest["files"].items():
    assert sha(frozen / name) == digest

xs, gs = specs(manifest["feature_groups"])

d = pd.read_csv(data_path)
d = d.sort_values(["w", "src", "dst"])

assert not d.duplicated(
    ["w", "src", "dst"]
).any()

torch.set_num_threads(8)

windows = sorted(d.w.unique())

rng = np.random.default_rng(2026)

selected = sorted(
    rng.choice(
        windows,
        size=min(200, len(windows)),
        replace=False
    )
)

frames = {
    int(w): g.reset_index(drop=True)
    for w, g in d[d.w.isin(selected)].groupby("w")
}

pd.DataFrame({
    "window": list(frames),
    "nodes": [len(g) for g in frames.values()]
}).to_csv(
    out / "benchmark_windows.csv",
    index=False
)

devices = ["cpu"]

if torch.cuda.is_available():
    devices.append("cuda")

rows = []
complexity = []

def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()

def benchmark(fn, device):
    sync(device)
    t0 = time.perf_counter()
    result = fn()
    sync(device)
    elapsed = (time.perf_counter()-t0)*1000

    assert np.isfinite(result).all()

    return elapsed

for device in devices:

    for seed in [42, 43, 44]:

        predictors = {}
        preparers = {}

        # XGBoost models
        for name, features in xs.items():

            path = frozen / f"{seed}_{name}.json"

            model = xgb.Booster()
            model.load_model(str(path))

            model.set_param({
                "device": device,
                "nthread": 8
            })

            def prep(frame, features=features):
                return xgb.DMatrix(
                    frame[features],
                    nthread=8
                )

            def predict(batch, model=model):
                return model.predict(batch)

            preparers[name] = prep
            predictors[name] = predict

            complexity.append({
                "device": device,
                "seed": seed,
                "model": name,
                "parameters": None,
                "trees": len(model.get_dump()),
                "model_bytes": path.stat().st_size
            })

        # Neural models
        for name, (mode, features) in gs.items():

            path = frozen / f"{seed}_{name}.pt"
            normpath = frozen / f"{seed}_{name}.npz"

            with np.load(normpath) as norm:
                mean = norm["mean"].copy()
                std = norm["std"].copy()

            model = Model(len(features)).to(device)

            model.load_state_dict(
                torch.load(
                    path,
                    map_location=device,
                    weights_only=True
                )
            )

            model.eval()

            def prep(
                frame,
                features=features,
                mean=mean,
                std=std,
                mode=mode,
                seed=seed
            ):

                _, batches = pack(
                    frame,
                    features,
                    mean,
                    std,
                    mode,
                    seed+200
                )

                return batches

            def predict(
                batches,
                model=model,
                mode=mode,
                device=device
            ):

                result = []

                with torch.inference_mode():

                    for x, g, n, _ in batches:

                        p = torch.sigmoid(
                            model(
                                x.to(device),
                                g.to(device),
                                n,
                                mode
                            )
                        )

                        result.append(
                            p.cpu().numpy()
                        )

                return np.concatenate(result)

            preparers[name] = prep
            predictors[name] = predict

            complexity.append({
                "device": device,
                "seed": seed,
                "model": name,
                "parameters": sum(
                    p.numel()
                    for p in model.parameters()
                ),
                "trees": None,
                "model_bytes": (
                    path.stat().st_size
                    + normpath.stat().st_size
                )
            })

        # Fixed hybrid M3 + G1
        def hybrid_prep(
            frame,
            pm=preparers["M3_full"],
            pg=preparers["G1_real_peer"]
        ):
            return pm(frame), pg(frame)

        def hybrid_predict(
            batches,
            pm=predictors["M3_full"],
            pg=predictors["G1_real_peer"]
        ):
            return (
                .5*pm(batches[0])
                + .5*pg(batches[1])
            )

        preparers["H1_fixed"] = hybrid_prep
        predictors["H1_fixed"] = hybrid_predict

        for name in predictors:

            prepare = preparers[name]
            predict = predictors[name]

            first = next(iter(frames.values()))

            warm = prepare(first)

            for _ in range(10):
                predict(warm)

            sync(device)

            for w, frame in frames.items():

                sync(device)

                t0 = time.perf_counter()
                batch = prepare(frame)
                sync(device)

                prep_ms = (
                    time.perf_counter()-t0
                )*1000

                peak = None

                if (
                    device == "cuda"
                    and name.startswith("G")
                ):
                    before = (
                        torch.cuda.memory_allocated()
                    )

                    torch.cuda.reset_peak_memory_stats()

                inference_ms = benchmark(
                    lambda: predict(batch),
                    device
                )

                if (
                    device == "cuda"
                    and name.startswith("G")
                ):
                    peak = max(
                        0,
                        torch.cuda.max_memory_allocated()
                        - before
                    ) / 1024**2

                rows.append({
                    "device": device,
                    "seed": seed,
                    "model": name,
                    "window": w,
                    "nodes": len(frame),
                    "preparation_ms": prep_ms,
                    "inference_ms": inference_ms,
                    "total_ms": (
                        prep_ms + inference_ms
                    ),
                    "torch_peak_extra_MiB": peak
                })

            print(
                "Completed:",
                device, seed, name,
                flush=True
            )

r = pd.DataFrame(rows)

r.to_csv(
    out / "latency_raw.csv",
    index=False
)

pd.DataFrame(complexity).to_csv(
    out / "model_complexity.csv",
    index=False
)

summaries = []

for (device, model, seed), g in r.groupby(
    ["device", "model", "seed"]
):

    summaries.append({
        "device": device,
        "model": model,
        "seed": seed,
        "windows": len(g),
        "mean_nodes": g.nodes.mean(),
        "preparation_mean_ms":
            g.preparation_ms.mean(),
        "inference_mean_ms":
            g.inference_ms.mean(),
        "total_mean_ms":
            g.total_ms.mean(),
        "total_p50_ms":
            g.total_ms.quantile(.50),
        "total_p95_ms":
            g.total_ms.quantile(.95),
        "total_p99_ms":
            g.total_ms.quantile(.99),
        "total_max_ms":
            g.total_ms.max(),
        "nodes_per_second":
            1000*g.nodes.sum()/g.total_ms.sum(),
        "peak_extra_MiB":
            g.torch_peak_extra_MiB.max(),
        "deadline_pass_rate":
            (g.total_ms < 10000).mean()
    })

s = pd.DataFrame(summaries)

s.to_csv(
    out / "latency_by_seed.csv",
    index=False
)

summary = s.groupby(
    ["device", "model"]
).mean(numeric_only=True).drop(
    columns="seed"
)

summary.to_csv(
    out / "latency_summary.csv"
)

environment = {
    "platform": platform.platform(),
    "torch": torch.__version__,
    "xgboost": xgb.__version__,
    "cpu_threads": torch.get_num_threads(),
    "gpu": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None
    ),
    "source_day": "2019-04-26",
    "window_sample_seed": 2026,
    "window_count": len(frames),
    "warmup": 10,
    "freeze_sha256": sha(frozen / "freeze.json")
}

(out / "environment.json").write_text(
    json.dumps(environment, indent=2)
)

print("\n=== MODEL COMPLEXITY ===")

print(
    pd.DataFrame(complexity)
    .query("device == 'cpu' and seed == 42")
    .to_string(index=False)
)

print("\n=== COMPUTATIONAL BENCHMARK ===")

print(summary.round(6).to_string())

print("\n=== ENVIRONMENT ===")
print(json.dumps(environment, indent=2))

print("\nSaved:", out)
