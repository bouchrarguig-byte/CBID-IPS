#!/usr/bin/env python3
"""Read-only decision audit for prior CBID-IPS DDoS feasibility V3.1 outputs.

Does NOT read raw traffic, train models, or access the sealed 26 April test.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

FROZEN_MIN_WINDOWS = 200
FROZEN_MIN_HEADROOM = .10


def fail(message):
    raise SystemExit('ERREUR: ' + message)


def num(row, key):
    val = row.get(key, '')
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path.home()/'Downloads'/'CBID_31617_DDoS_25Apr_feasibility_v3_1', help='Existing V3.1 feasibility output directory')
    args = parser.parse_args()
    folder = args.results.expanduser().resolve()
    verdict_path = folder / 'feasibility_verdict.json'
    support_path = folder / 'split_support_and_sources.csv'
    if not verdict_path.exists():
        fail(f'{verdict_path} introuvable. Ne relancez pas un entrainement pour ce seul audit; retrouvez le dossier V3.1, ou indiquez-le avec --results.')
    if not support_path.exists():
        fail(f'{support_path} introuvable. Le contrôle des sources indépendantes est impossible.')
    data = json.loads(verdict_path.read_text(encoding='utf-8'))
    with support_path.open(newline='',encoding='utf-8-sig') as f:
        support = list(csv.DictReader(f))
    b_rows = [r for r in support if r.get('period')=='B_feasibility']
    if len(b_rows)!=1:
        fail('Le bilan doit contenir exactement une ligne B_feasibility.')
    b=b_rows[0]
    n_w = int(data['B_coordinated_windows'])
    npos = int(data['B_coordinated_positive_source_windows'])
    missed = int(data['missed_by_context'])
    headroom = float(data['headroom'])
    if npos <= 0 or missed < 0 or missed > npos:
        fail('Comptages de positifs coordonnés incohérents.')
    if not math.isclose(headroom, missed/npos, abs_tol=1e-7):
        fail('Taux de marge incohérent avec les effectifs: vérifier les fichiers.')
    b_w = num(b,'coordinated_dest_windows_ge2')
    if b_w is not None and int(b_w)!=n_w:
        fail('Nombre de fenêtres coordonnées contradictoire entre les fichiers.')
    n_sets = num(b,'unique_attacking_source_sets_in_coord')
    n_ip = num(b,'unique_DDoS_source_IPs')
    max_share = num(b,'share_coord_windows_largest_source_set')
    pass_windows = n_w>=FROZEN_MIN_WINDOWS
    pass_headroom = headroom>=FROZEN_MIN_HEADROOM
    go = pass_windows and pass_headroom
    verdict = 'GO_POUR_ANALYSE_SUPPLEMENTAIRE_NON_VALIDATION' if go else 'NO_GO_ARRET_DE_CETTE_PISTE'
    warnings=[]
    if n_sets is None or n_ip is None:
        warnings.append('La diversité des IP et des groupes attaquants ne peut pas être vérifiée.')
    if max_share is not None and max_share>=.8:
        warnings.append('Au moins 80% des fenêtres ont le même ensemble de sources attaquantes : concentration très forte.')
    warnings.append('Différents ensembles d’IP ne prouvent pas des campagnes d’attaque indépendantes.')
    warnings.append('La période B est exploratoire, pas une validation aveugle ; la tâche DDoS a été sélectionnée après examen des étiquettes.')
    warnings.append('Le 26 avril 2019 demeure exclu et n’a pas été consulté par cet outil.')
    output={
        'audit':'CBID-IPS #31617 – gate de nouveauté DDoS, lecture seule',
        'result_directory':str(folder), 'period':'B, 25 avril 2019, exploration',
        'frozen_rules':{'min_coordinated_destination_windows':FROZEN_MIN_WINDOWS,'min_fraction_of_context_baseline_misses':FROZEN_MIN_HEADROOM},
        'observed':{'coordinated_destination_windows':n_w,'coordinated_positive_source_windows':npos,
            'missed_by_contextual_XGBoost':missed,'fraction_missed':headroom,
            'unique_attacking_source_sets_in_coord':int(n_sets) if n_sets is not None else None,
            'unique_DDoS_source_IPs':int(n_ip) if n_ip is not None else None,
            'share_windows_largest_attacking_set':max_share},
        'criteria_met':{'count_of_coordinated_windows':pass_windows,'baseline_headroom':pass_headroom},
        'verdict':verdict,'limitations':warnings}
    folder.joinpath('novelty_gate_result.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    share_text=(f'{max_share:.1%}' if max_share is not None else 'non disponible')
    report=f'''# CBID-IPS #31617 — Audit de faisabilité de la nouveauté inter-hôtes\n\n**Décision : {verdict}.**\n\nCet audit relit les sorties déjà enregistrées de l'expérience DDoS exploratoire du 25 avril 2019. Il n'entraîne aucun modèle et ne lit aucune donnée du 26 avril.\n\n- Fenêtres destination multi-sources positives : **{n_w:,}** (seuil préalablement fixé : {FROZEN_MIN_WINDOWS:,}).\n- Observations positives coordonnées : **{npos:,}**.\n- Manquées par XGBoost contextuel : **{missed:,}** (**{headroom:.2%}**, seuil fixé : {FROZEN_MIN_HEADROOM:.0%}).\n- Ensembles de sources attaquantes distincts observés : **{int(n_sets) if n_sets is not None else 'non disponible'}**.\n- IP sources DDoS distinctes : **{int(n_ip) if n_ip is not None else 'non disponible'}**.\n- Part des fenêtres du plus fréquent ensemble de sources : **{share_text}**.\n'''
    report+='\n## Limites\n\n'+'\n'.join(f'- {x}' for x in warnings)+'\n\n'
    report += ('**Décision pratique : arrêter cette piste expérimentale et finaliser le manuscrit sans revendication de gain inter-hôtes démontré.**\n' if not go else '**Décision pratique : seul un audit des campagnes indépendantes serait justifié; GO ne prouve aucun gain du GNN.**\n')
    folder.joinpath('novelty_gate_report.md').write_text(report,encoding='utf-8')
    print(report)
    print('SORTIES:',folder/'novelty_gate_result.json',folder/'novelty_gate_report.md')

if __name__=='__main__':
    main()
