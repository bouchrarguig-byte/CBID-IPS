#!/usr/bin/env python3
"""Audit/extract NF-ToN-IoT-v3 real host-to-destination scanning traffic.

Read ZIP in streaming mode; do not execute original project code or modify input.
Output exploratory aggregated CSVs for a subsequent leakage-controlled evaluation.
All label-derived counts are explicitly marked AUDIT_ONLY and MUST NOT be used
as model features. No training/testing split or classifier is performed here.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Set

TARGET_DESTINATIONS = ("192.168.1.31", "192.168.1.32")
DEFAULT_ARCHIVE = Path.home() / "Downloads" / "02934b58528a226b_NFV3DATA-A11964_A11964.zip"


def number(x: str) -> float:
    try:
        n = float(x)
        return n if -1.0e18 < n < 1.0e18 else 0.0
    except (ValueError, TypeError):
        return 0.0


@dataclass
class Acc:
    n: int = 0
    in_bytes: float = 0.0
    out_bytes: float = 0.0
    in_pkts: float = 0.0
    out_pkts: float = 0.0
    duration_ms: float = 0.0
    tcp_flows: int = 0
    udp_flows: int = 0
    dst_ports: Set[str] = field(default_factory=set)
    attack_flows: int = 0
    benign_flows: int = 0

    def add(self, row: dict[str, str], malicious: bool) -> None:
        self.n += 1
        self.in_bytes += number(row.get("IN_BYTES", "0"))
        self.out_bytes += number(row.get("OUT_BYTES", "0"))
        self.in_pkts += number(row.get("IN_PKTS", "0"))
        self.out_pkts += number(row.get("OUT_PKTS", "0"))
        self.duration_ms += max(0, number(row.get("FLOW_DURATION_MILLISECONDS", "0")))
        proto = row.get("PROTOCOL", "").strip()
        self.tcp_flows += proto in ("6", "TCP", "tcp")
        self.udp_flows += proto in ("17", "UDP", "udp")
        port = row.get("L4_DST_PORT", "").strip()
        if port:
            self.dst_ports.add(port)
        self.attack_flows += int(malicious)
        self.benign_flows += int(not malicious)


def iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--output", type=Path, default=Path.home() / "Downloads" / "CBID_NFToN_Scanning_Audit")
    ap.add_argument("--day", default="2019-04-24", help="UTC day YYYY-MM-DD")
    ap.add_argument("--window", type=int, default=10, help="nonoverlapping window size in seconds")
    ap.add_argument("--time-field", choices=("start", "end"), default="end",
                    help="end makes aggregate features available at decision time")
    args = ap.parse_args()
    if args.window < 1:
        ap.error("--window must be positive")
    try:
        d = date.fromisoformat(args.day)
    except ValueError:
        ap.error("Invalid --day (expected YYYY-MM-DD)")
    begin_ms = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = begin_ms + 86400 * 1000
    timestamp_col = ("FLOW_END_MILLISECONDS" if args.time_field == "end" else "FLOW_START_MILLISECONDS")
    window_ms = args.window * 1000
    src_aggs: dict[tuple[int, str, str], Acc] = {}
    counters = Counter()
    seen_classes = Counter()
    selected_dst = set(TARGET_DESTINATIONS)
    print("Archive:", args.archive, flush=True)
    print("Destinations:", ", ".join(TARGET_DESTINATIONS), flush=True)
    print("Day UTC:", d.isoformat(), "| bin:", args.window, "s | timestamp:", timestamp_col, flush=True)

    with zipfile.ZipFile(args.archive) as z:
        csv_filename = next((n for n in z.namelist() if n.endswith("/data/NF-ToN-IoT-v3.csv")), None)
        if csv_filename is None:
            raise RuntimeError("NF-ToN-IoT-v3.csv not found in ZIP")
        with z.open(csv_filename) as raw:
            stream = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="", errors="replace")
            reader = csv.DictReader(stream)
            required = {"Label", "Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", timestamp_col}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise RuntimeError("Missing expected columns: " + ", ".join(sorted(missing)))
            for row in reader:
                counters["all_rows"] += 1
                if counters["all_rows"] % 3_000_000 == 0:
                    print(f"Read {counters['all_rows']:,} rows; selected {counters['selected_rows']:,}", flush=True)
                dst = row["IPV4_DST_ADDR"].strip()
                if dst not in selected_dst:
                    continue
                counters["matching_destination_all_days"] += 1
                try:
                    t = int(float(row[timestamp_col]))
                except (ValueError, TypeError, OverflowError):
                    counters["bad_timestamps"] += 1
                    continue
                if not (begin_ms <= t < end_ms):
                    continue
                label = row["Label"].strip()
                attack = row["Attack"].strip().lower()
                seen_classes[(label, attack)] += 1
                # Restrict the experiment explicitly to a binary scanning/benign task.
                if label == "1" and attack == "scanning":
                    malicious = True
                elif label == "0" and attack == "benign":
                    malicious = False
                else:
                    counters["excluded_other_or_inconsistent_labels"] += 1
                    continue
                src = row["IPV4_SRC_ADDR"].strip()
                if not src:
                    counters["blank_source"] += 1
                    continue
                win = (t // window_ms) * window_ms
                key = win, dst, src
                if key not in src_aggs:
                    src_aggs[key] = Acc()
                src_aggs[key].add(row, malicious)
                counters["selected_rows"] += 1
    args.output.mkdir(parents=True, exist_ok=True)

    host_csv = args.output / f"host_window_{args.time_field}_{args.day}_{args.window}s.csv"
    graph_csv = args.output / f"destination_window_audit_{args.time_field}_{args.day}_{args.window}s.csv"
    graph_stats: dict[tuple[int, str], dict] = defaultdict(lambda: {
        "flow_count": 0, "scanning_flows": 0, "benign_flows": 0,
        "source_count": 0, "positive_hosts": 0, "benign_hosts": 0,
    })
    with host_csv.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow([
            "window_start_ms_META", "window_utc_META", "destination_ip_META", "source_ip_META",
            "n_flows", "sum_in_bytes", "sum_out_bytes", "sum_in_pkts", "sum_out_pkts",
            "mean_flow_duration_ms", "n_unique_dst_ports", "fraction_tcp", "fraction_udp",
            "attack_flows_AUDIT_ONLY", "benign_flows_AUDIT_ONLY", "y_scanning_host_window",
        ])
        for (win, dst, src), a in sorted(src_aggs.items()):
            wr.writerow([win, iso(win), dst, src, a.n, int(a.in_bytes), int(a.out_bytes),
                         int(a.in_pkts), int(a.out_pkts), round(a.duration_ms / a.n, 5),
                         len(a.dst_ports), round(a.tcp_flows / a.n, 7), round(a.udp_flows / a.n, 7),
                         a.attack_flows, a.benign_flows, int(a.attack_flows > 0)])
            g = graph_stats[(win, dst)]
            g["flow_count"] += a.n
            g["scanning_flows"] += a.attack_flows
            g["benign_flows"] += a.benign_flows
            g["source_count"] += 1
            g["positive_hosts"] += int(a.attack_flows > 0)
            g["benign_hosts"] += int(a.benign_flows > 0)

    with graph_csv.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["window_start_ms_META", "window_utc_META", "destination_ip_META",
                     "source_count", "positive_hosts_AUDIT_ONLY", "benign_hosts_AUDIT_ONLY",
                     "scanning_flows_AUDIT_ONLY", "benign_flows_AUDIT_ONLY", "total_flows"])
        for (win, dst), g in sorted(graph_stats.items()):
            wr.writerow([win, iso(win), dst, g["source_count"], g["positive_hosts"],
                         g["benign_hosts"], g["scanning_flows"], g["benign_flows"], g["flow_count"]])

    summaries: dict[str, dict] = {}
    for dst in TARGET_DESTINATIONS:
        windows = sorted((win, g) for (win, dd), g in graph_stats.items() if dd == dst)
        n = len(windows)
        positives = sum(g["positive_hosts"] > 0 for _, g in windows)
        negatives = n - positives
        mixed = sum(g["scanning_flows"] > 0 and g["benign_flows"] > 0 for _, g in windows)
        multi = sum(g["positive_hosts"] >= 2 for _, g in windows)
        multi5 = sum(g["positive_hosts"] >= 5 for _, g in windows)
        max_src = max((g["positive_hosts"] for _, g in windows), default=0)
        # Chronological thirds of the observed windows: preliminary feasibility only.
        ranges = []
        for i, j in ((0, n//3), (n//3, 2*n//3), (2*n//3, n)):
            sub = windows[i:j]
            ranges.append({"windows": len(sub), "positive_windows": sum(g["positive_hosts"] > 0 for _, g in sub),
                           "negative_windows": sum(g["positive_hosts"] == 0 for _, g in sub),
                           "start_utc": iso(sub[0][0]) if sub else None,
                           "end_utc": iso(sub[-1][0]) if sub else None})
        summaries[dst] = {"windows_total": n, "attack_windows": positives,
                          "benign_only_windows": negatives, "mixed_windows": mixed,
                          "windows_multi_source_ge2": multi, "windows_multi_source_ge5": multi5,
                          "max_simultaneous_scanning_sources": max_src,
                          "chronological_thirds_diagnostic_only": ranges}

    report = {
        "purpose": "Exploratory, NOT a finished leak-free validation or proof of coordination",
        "archive": str(args.archive), "date_utc": str(d), "time_field": timestamp_col,
        "window_seconds": args.window, "destinations": TARGET_DESTINATIONS,
        "counters": dict(counters),
        "observed_labels_at_selected_destinations_day": {f"{l} / {a}": n for (l, a), n in seen_classes.items()},
        "summaries": summaries,
        "leakage_warning": "Do not use *_META (host identity or UTC time) or *_AUDIT_ONLY (label-derived) as predictive features. Use labels only as targets. Split by blocked time/host before all fitting and tuning; avoid future information at prediction time.",
    }
    json_path = args.output / f"summary_{args.time_field}_{args.day}_{args.window}s.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n===== RESULTS =====")
    print("Total records:", f"{counters['all_rows']:,}", "| scoped records:", f"{counters['selected_rows']:,}")
    for dst, s in summaries.items():
        print("\nDESTINATION", dst)
        print("windows:", s["windows_total"], "scanning:", s["attack_windows"],
              "benign-only:", s["benign_only_windows"], "mixed:", s["mixed_windows"])
        print("multi-source >=2:", s["windows_multi_source_ge2"],
              ">=5:", s["windows_multi_source_ge5"],
              "max:", s["max_simultaneous_scanning_sources"])
        for i, p in enumerate(s["chronological_thirds_diagnostic_only"], 1):
            print(f"chronological third {i}: windows={p['windows']} positive={p['positive_windows']} benign_only={p['negative_windows']} from={p['start_utc']} to={p['end_utc']}")
    print("\nFiles created:", host_csv, graph_csv, json_path, sep="\n - ")
    print("\nNOTE: descriptive extraction only; no train/test model or data cleaning yet.")


if __name__ == "__main__":
    main()
