#!/usr/bin/env python3
"""Read-only two-pass audit for real concurrent DDoS/scanning sources in NF-ToN-IoT-v3.

Does not extract the ZIP, train models or write files. Requires Python 3 standard library.
"""
from __future__ import annotations

import csv
import io
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ATTACKS = ("ddos", "scanning")
BIN_S = 10
MAX_PER_ATTACK = 4  # two by volume; two by distinct sources


def stream_csv(z: zipfile.ZipFile, member: str):
    with z.open(member) as source:
        with io.TextIOWrapper(source, encoding="utf-8-sig", errors="replace", newline="") as f:
            yield from csv.DictReader(f)


def get_bin(s: str):
    try:
        n = int(s)
        # NF-ToN-IoT-v3 timestamps are milliseconds since epoch.
        if n < 1_000_000_000_000 or n > 4_000_000_000_000:
            return None
        return n // (1000 * BIN_S)
    except (ValueError, TypeError):
        return None


def utc_bin(b: int) -> str:
    return datetime.fromtimestamp(b * BIN_S, tz=timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    if len(sys.argv) > 1:
        archive = Path(sys.argv[1]).expanduser()
    else:
        archive = Path.home() / "Downloads/02934b58528a226b_NFV3DATA-A11964_A11964.zip"
    if not archive.is_file():
        print("Archive introuvable:", archive, file=sys.stderr)
        return 1

    print("Archive:", archive, flush=True)
    print("Mode: lecture seule, analyse par fenêtres de 10 secondes UTC", flush=True)
    with zipfile.ZipFile(archive) as z:
        members = [name for name in z.namelist() if name.endswith("/NF-ToN-IoT-v3.csv") or name == "NF-ToN-IoT-v3.csv"]
        if len(members) != 1:
            print("Impossible de trouver un unique NF-ToN-IoT-v3.csv", file=sys.stderr)
            return 2
        member = members[0]

        pair_count = Counter()
        pair_sources = defaultdict(set)
        bad_labels = 0
        total = 0
        print("\nPASSAGE 1/2 : destinations et sources malveillantes...", flush=True)
        for row in stream_csv(z, member):
            total += 1
            attack = row["Attack"].strip().lower()
            if attack not in ATTACKS:
                continue
            if row["Label"].strip() != "1":
                bad_labels += 1
                continue
            dest = row["IPV4_DST_ADDR"].strip()
            src = row["IPV4_SRC_ADDR"].strip()
            if not dest or not src:
                continue
            key = (attack, dest)
            pair_count[key] += 1
            pair_sources[key].add(src)

        chosen = set()
        print("Total des flux:", f"{total:,}")
        print("Flux ddos/scanning avec étiquette incohérente:", bad_labels)
        for attack in ATTACKS:
            relevant = [(key, n) for key, n in pair_count.items() if key[0] == attack]
            by_volume = sorted(relevant, key=lambda x: x[1], reverse=True)
            multi = sorted([x for x in relevant if len(pair_sources[x[0]]) >= 2],
                           key=lambda x: (len(pair_sources[x[0]]), x[1]), reverse=True)
            print(f"\n{attack.upper()} — {sum(n for _, n in relevant):,} flux sur {len(relevant):,} destinations")
            print("Top destinations par volume:")
            for (a, dst), n in by_volume[:6]:
                print(f"  {dst:<19} flux={n:>10,}  sources_distinctes={len(pair_sources[(a, dst)])}")
            print("Top destinations avec >=2 sources:")
            for (a, dst), n in multi[:6]:
                print(f"  {dst:<19} flux={n:>10,}  sources_distinctes={len(pair_sources[(a, dst)])}")
            chosen.update(key for key, _ in by_volume[:2])
            chosen.update(key for key, _ in multi[:2])

        if not chosen:
            print("Aucune destination candidate; arrêt.")
            return 0
        targets = {dst for a, dst in chosen}
        print(f"\nPASSAGE 2/2 : fenêtres de {BIN_S}s pour {len(targets)} destinations...", flush=True)
        # For each (destination, ten-second bucket) collect benign and DDoS/scan source IDs.
        windows = {}  # (dst, b) -> [benign_count, benign_src_set, attack_counts, attack_sources]
        bad_time = 0
        for row in stream_csv(z, member):
            dst = row["IPV4_DST_ADDR"].strip()
            if dst not in targets:
                continue
            b = get_bin(row["FLOW_START_MILLISECONDS"])
            if b is None:
                bad_time += 1
                continue
            k = (dst, b)
            if k not in windows:
                windows[k] = [0, set(), Counter(), defaultdict(set)]
            state = windows[k]
            src = row["IPV4_SRC_ADDR"].strip()
            label = row["Label"].strip()
            attack = row["Attack"].strip().lower()
            if label == "0":
                state[0] += 1
                state[1].add(src)
            elif label == "1":
                state[2][attack] += 1
                state[3][attack].add(src)

        print("Fenêtres totales stockées:", f"{len(windows):,}", "| horodatages invalides:", bad_time)
        print("\n========== FAISABILITÉ MULTI-SOURCE RÉELLE ==========")
        for attack, dst in sorted(chosen):
            active = []
            for (d, b), (n_benign, s_benign, a_counts, a_sources) in windows.items():
                if d != dst or a_counts[attack] == 0:
                    continue
                active.append((b, a_counts[attack], len(a_sources[attack]), n_benign,
                               len(s_benign)))
            if not active:
                continue
            print(f"\n{attack.upper()} → {dst} | {pair_count[(attack,dst)]:,} flux attack | "
                  f"{len(pair_sources[(attack,dst)])} sources globales")
            print(f"  Fenêtres de {BIN_S}s avec attaque: {len(active):,}; "
                  f"avec >=2 sources: {sum(x[2] >= 2 for x in active):,}; "
                  f">=3: {sum(x[2] >= 3 for x in active):,}; "
                  f">=5: {sum(x[2] >= 5 for x in active):,}")
            print(f"  Maximum de sources simultanées: {max(x[2] for x in active)}; "
                  f"fenêtres avec trafic bénin vers même destination: {sum(x[3] > 0 for x in active):,}")
            print("  Exemples de fenêtres multi-source les plus actives:")
            for b, n_attack, n_src, n_benign, n_benign_src in sorted(
                active, key=lambda x: (x[2], x[1]), reverse=True
            )[:5]:
                print(f"    {utc_bin(b)}  attaques={n_attack:,} "
                      f"sources_attack={n_src}  flux_bénins={n_benign:,} "
                      f"sources_bénignes={n_benign_src}")

    print("\nInterprétation: seules les fenêtres avec >=2 sources malveillantes "
          "vers une même destination attestent la simultanéité multi-source "
          "au niveau des étiquettes de flux; elles ne prouvent pas à elles seules "
          "que chaque IP source est un objet IoT compromis.")
    print("Fin — aucune modification de données, aucun entraînement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
