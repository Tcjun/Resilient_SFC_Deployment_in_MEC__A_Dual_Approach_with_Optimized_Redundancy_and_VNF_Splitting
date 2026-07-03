#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-placement correlated-failure stress test for accepted SFC placements."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import pandas as pd


def parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def build_failure_domains(edge_csv: Path, region: str, domain_count: int) -> Dict[int, int]:
    df = pd.read_csv(edge_csv)
    df = df[df["Region"] == region].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No cloudlets found for region={region}")
    ordered = df.sort_values(["Longitude", "Latitude"]).index.tolist()
    mapping: Dict[int, int] = {}
    n = len(ordered)
    for rank, cloudlet_id in enumerate(ordered):
        domain = min(int(rank * int(domain_count) / max(n, 1)), int(domain_count) - 1)
        mapping[int(cloudlet_id)] = int(domain)
    return mapping


def failed_domain_sets(domain_count: int, q: float) -> Iterable[tuple[Set[int], float]]:
    domains = list(range(int(domain_count)))
    for mask in range(1 << int(domain_count)):
        failed = {d for d in domains if mask & (1 << d)}
        fail_n = len(failed)
        weight = (float(q) ** fail_n) * ((1.0 - float(q)) ** (int(domain_count) - fail_n))
        yield failed, weight


def instance_survives(instance: Dict, failed_domains: Set[int], domain_of: Dict[int, int]) -> bool:
    cloudlet = int(instance["cloudlet"])
    return int(domain_of[cloudlet]) not in failed_domains


def stressed_request_reliability(entry: Dict, failed_domains: Set[int], domain_of: Dict[int, int]) -> float:
    instances = list(entry.get("instances", []))
    base_reli = {int(k): float(v) for k, v in dict(entry.get("vnf_base_reliability", {})).items()}
    total = 1.0
    for v in [int(x) for x in entry.get("chain", [])]:
        r = float(base_reli.get(v, 0.0))
        stage_instances = [inst for inst in instances if int(inst["vnf"]) == v]
        active_instances = [inst for inst in stage_instances if int(inst["is_backup"]) == 0]
        backup_instances = [inst for inst in stage_instances if int(inst["is_backup"]) != 0]

        surviving_active = [inst for inst in active_instances if instance_survives(inst, failed_domains, domain_of)]
        if len(surviving_active) <= 0:
            active_rel = 0.0
        elif any(int(inst.get("replica", -1)) >= 0 for inst in active_instances):
            active_rel = 1.0 - ((1.0 - r) ** len(surviving_active))
        else:
            active_rel = r

        fail_prob = 1.0 - active_rel
        for inst in backup_instances:
            backup_rel = r if instance_survives(inst, failed_domains, domain_of) else 0.0
            fail_prob *= (1.0 - backup_rel)
        stage_rel = 1.0 - fail_prob
        total *= stage_rel
    return float(total)


def stress_trace(trace_path: Path, domain_of: Dict[int, int], domain_count: int, q: float) -> Dict:
    with trace_path.open("r", encoding="utf-8") as f:
        trace = json.load(f)
    total_accepted = len(trace)
    if total_accepted == 0:
        return {
            "total_accepted": 0,
            "expected_target_satisfaction_ratio": 0.0,
            "expected_reliability": 0.0,
            "expected_reliability_loss": 0.0,
        }

    expected_pass = 0.0
    expected_reli_sum = 0.0
    base_reli_sum = sum(float(entry.get("realized_reliability", 0.0)) for entry in trace)

    for failed, weight in failed_domain_sets(domain_count, q):
        pass_count = 0
        reli_sum = 0.0
        for entry in trace:
            reli = stressed_request_reliability(entry, failed, domain_of)
            reli_sum += reli
            if reli + 1e-12 >= float(entry.get("required_reliability", 1.0)):
                pass_count += 1
        expected_pass += float(weight) * (pass_count / float(total_accepted))
        expected_reli_sum += float(weight) * (reli_sum / float(total_accepted))

    base_avg = base_reli_sum / float(total_accepted)
    return {
        "total_accepted": int(total_accepted),
        "expected_target_satisfaction_ratio": float(expected_pass),
        "expected_reliability": float(expected_reli_sum),
        "expected_reliability_loss": float(base_avg - expected_reli_sum),
    }


def ci95(series: pd.Series) -> float:
    n = int(series.count())
    if n <= 1:
        return 0.0
    return float(1.96 * series.std(ddof=1) / math.sqrt(n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", default="paper_data/workload_sharing_sensitivity/raw_metrics_long.csv")
    parser.add_argument("--edge_csv", default="Edge_devices.csv")
    parser.add_argument("--out_dir", default="paper_data/correlated_failure_stress")
    parser.add_argument("--domain_count", type=int, default=6)
    parser.add_argument("--domain_failure_probs", default="0,0.01,0.03,0.05")
    parser.add_argument("--share_filter", default="", help="Optional share_tag filter, e.g., 0p5_0p5.")
    args = parser.parse_args()

    metrics_csv = Path(args.metrics_csv).resolve()
    edge_csv = Path(args.edge_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(metrics_csv)
    raw = raw[raw["placement_trace_path"].notna() & (raw["placement_trace_path"].astype(str) != "")]
    if args.share_filter:
        raw = raw[(raw["method"] == "MDCA") | (raw["share_tag"] == args.share_filter)]
    probs = parse_csv_floats(args.domain_failure_probs)

    domain_cache: Dict[str, Dict[int, int]] = {}
    rows: List[Dict] = []
    for _, row in raw.iterrows():
        region = str(row["region"])
        if region not in domain_cache:
            domain_cache[region] = build_failure_domains(edge_csv, region, int(args.domain_count))
        trace_path = Path(str(row["placement_trace_path"]))
        for q in probs:
            stressed = stress_trace(trace_path, domain_cache[region], int(args.domain_count), float(q))
            rows.append({
                "region": region,
                "fixed_reliability": float(row["fixed_reliability"]),
                "seed": int(row["seed"]),
                "method": str(row["method"]),
                "share": str(row.get("share", "")),
                "share_tag": str(row.get("share_tag", "")),
                "domain_count": int(args.domain_count),
                "domain_failure_prob": float(q),
                **stressed,
            })

    stress_raw = pd.DataFrame(rows)
    raw_path = out_dir / "raw_stress_metrics.csv"
    stress_raw.to_csv(raw_path, index=False)

    metric_cols = [
        "total_accepted",
        "expected_target_satisfaction_ratio",
        "expected_reliability",
        "expected_reliability_loss",
    ]
    group_cols = ["region", "fixed_reliability", "method", "share", "share_tag", "domain_count", "domain_failure_prob"]
    summary = stress_raw.groupby(group_cols)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in summary.columns]
    for col in metric_cols:
        summary[f"{col}_ci95"] = stress_raw.groupby(group_cols)[col].apply(ci95).reset_index(drop=True)
    summary_path = out_dir / "summary_stress_metrics.csv"
    summary.to_csv(summary_path, index=False)

    config = {
        "metrics_csv": str(metrics_csv),
        "edge_csv": str(edge_csv),
        "domain_count": int(args.domain_count),
        "domain_failure_probs": probs,
        "share_filter": args.share_filter,
    }
    (out_dir / "stress_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Saved raw stress: {raw_path}")
    print(f"Saved summary stress: {summary_path}")
    print(f"Saved config: {out_dir / 'stress_config.json'}")


if __name__ == "__main__":
    main()
