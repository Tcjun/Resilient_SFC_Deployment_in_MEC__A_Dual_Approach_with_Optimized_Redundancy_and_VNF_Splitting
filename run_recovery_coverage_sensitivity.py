#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run recovery-coverage sensitivity for split active-pool reliability."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def tag_float(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p")


def run_cmd(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError("Command failed:\n" + " ".join(cmd) + "\n\n" + proc.stdout)


def read_metrics(out_dir: Path, region: str, seed: int) -> Dict:
    path = out_dir / f"metrics_{region}_mdca_seed{seed}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ci95(series: pd.Series) -> float:
    n = int(series.count())
    if n <= 1:
        return 0.0
    return float(1.96 * series.std(ddof=1) / math.sqrt(n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_script", default="online_mdca_sim_delay_split_strict.py")
    parser.add_argument("--edge_csv", default="Edge_devices.csv")
    parser.add_argument("--out_root", default="paper_data/recovery_coverage_sensitivity")
    parser.add_argument("--regions", default="Alabama,Arizona")
    parser.add_argument("--fixed_reliabilities", default="0.999")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--etas", default="0.9,0.95,0.99,1.0")
    parser.add_argument("--time_slots", type=int, default=120)
    parser.add_argument("--arrival_rate", type=float, default=12.0)
    parser.add_argument("--mean_service_time", type=float, default=9.0)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--cap_scale", type=float, default=0.2)
    parser.add_argument("--backhaul_knn_k", type=int, default=4)
    parser.add_argument("--backhaul_bw_scale", type=float, default=0.5)
    parser.add_argument("--max_backup_level", type=int, default=2)
    parser.add_argument("--p_split", type=float, default=0.3)
    parser.add_argument("--split_replica_count", type=int, default=2)
    parser.add_argument("--split_workload_shares", default="0.5,0.5")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    sim_script = Path(args.sim_script).resolve()
    edge_csv = Path(args.edge_csv).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    regions = [x.strip() for x in args.regions.split(",") if x.strip()]
    reliabilities = parse_csv_floats(args.fixed_reliabilities)
    seeds = parse_csv_ints(args.seeds)
    etas = parse_csv_floats(args.etas)
    rows: List[Dict] = []
    commands: List[str] = []

    for eta in etas:
        if not (0.0 <= float(eta) <= 1.0):
            raise ValueError("--etas values must be between 0 and 1")

    for region in regions:
        for reliability in reliabilities:
            reliability_tag = f"R{tag_float(reliability)}"
            for seed in seeds:
                mdca_out = out_root / region / reliability_tag / "MDCA" / f"seed{seed}"
                mdca_out.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable, str(sim_script),
                    "--edge_csv", str(edge_csv),
                    "--region", region,
                    "--time_slots", str(args.time_slots),
                    "--arrival_rate", str(args.arrival_rate),
                    "--mean_service_time", str(args.mean_service_time),
                    "--k", str(args.k),
                    "--algo", "mdca",
                    "--cap_scale", str(args.cap_scale),
                    "--backhaul_knn_k", str(args.backhaul_knn_k),
                    "--backhaul_bw_scale", str(args.backhaul_bw_scale),
                    "--fixed_reliability", f"{reliability:.6g}",
                    "--max_backup_level", str(args.max_backup_level),
                    "--seed", str(seed),
                    "--out_dir", str(mdca_out),
                ]
                commands.append(" ".join(cmd))
                if not args.dry_run:
                    run_cmd(cmd)
                    metrics = read_metrics(mdca_out, region, seed)
                    metrics.update({
                        "method": "MDCA",
                        "recovery_coverage": "none",
                        "coverage_tag": "none",
                        "fixed_reliability": reliability,
                    })
                    rows.append(metrics)

                for eta in etas:
                    eta_tag = tag_float(float(eta))
                    out_dir = out_root / region / reliability_tag / f"DMDCA_eta{eta_tag}" / f"seed{seed}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        sys.executable, str(sim_script),
                        "--edge_csv", str(edge_csv),
                        "--region", region,
                        "--time_slots", str(args.time_slots),
                        "--arrival_rate", str(args.arrival_rate),
                        "--mean_service_time", str(args.mean_service_time),
                        "--k", str(args.k),
                        "--algo", "mdca",
                        "--cap_scale", str(args.cap_scale),
                        "--backhaul_knn_k", str(args.backhaul_knn_k),
                        "--backhaul_bw_scale", str(args.backhaul_bw_scale),
                        "--fixed_reliability", f"{reliability:.6g}",
                        "--max_backup_level", str(args.max_backup_level),
                        "--seed", str(seed),
                        "--diversity",
                        "--p_split", str(args.p_split),
                        "--explicit_split_replicas",
                        "--split_replica_count", str(args.split_replica_count),
                        "--split_reliability_mode", "identical_active_pool",
                        "--split_recovery_coverage", f"{float(eta):.6g}",
                        "--split_workload_shares", str(args.split_workload_shares),
                        "--out_dir", str(out_dir),
                    ]
                    commands.append(" ".join(cmd))
                    if args.dry_run:
                        continue
                    run_cmd(cmd)
                    metrics = read_metrics(out_dir, region, seed)
                    metrics.update({
                        "method": "DMDCA",
                        "recovery_coverage": float(metrics.get("split_recovery_coverage", eta)),
                        "coverage_tag": eta_tag,
                        "fixed_reliability": reliability,
                    })
                    rows.append(metrics)

    (out_root / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    if args.dry_run:
        print(f"Dry run commands: {len(commands)}")
        return

    raw = pd.DataFrame(rows)
    raw_path = out_root / "raw_metrics_long.csv"
    raw.to_csv(raw_path, index=False)

    metric_cols = [
        "acceptance_ratio",
        "avg_cost_per_accepted",
        "p95_e2e_delay_accepted",
        "avg_e2e_delay_accepted",
        "capacity_rejection_ratio",
        "reliability_rejection_ratio",
        "qos_rejection_ratio",
        "avg_split_overhead_ratio_accepted",
        "p95_split_overhead_ratio_accepted",
        "avg_split_transmitted_bits_per_accepted",
    ]
    group_cols = ["region", "fixed_reliability", "method", "recovery_coverage", "coverage_tag"]
    agg = raw.groupby(group_cols)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in agg.columns]
    for col in metric_cols:
        agg[f"{col}_ci95"] = raw.groupby(group_cols)[col].apply(ci95).reset_index(drop=True)
    summary_path = out_root / "summary_metrics.csv"
    agg.to_csv(summary_path, index=False)

    fairness_rows = []
    for keys, group in raw.groupby(["region", "fixed_reliability", "seed"]):
        arrivals = group["arrivals_per_slot"].apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, list) else str(x))
        dmdca = group[group["method"] == "DMDCA"]
        fairness_rows.append({
            "region": keys[0],
            "fixed_reliability": keys[1],
            "seed": keys[2],
            "trace_hash_nunique": int(group["trace_hash"].nunique()),
            "arrivals_per_slot_key_nunique": int(arrivals.nunique()),
            "total_arrivals_nunique": int(group["total_arrivals"].nunique()),
            "dmdca_algo_seed_nunique": int(dmdca["algo_seed"].nunique()) if len(dmdca) > 0 else 0,
            "base_arrivals_consistent": bool(
                group["trace_hash"].nunique() == 1
                and arrivals.nunique() == 1
                and group["total_arrivals"].nunique() == 1
            ),
            "dmdca_random_stream_consistent_across_eta": bool(dmdca["algo_seed"].nunique() == 1) if len(dmdca) > 0 else True,
        })
    fairness_path = out_root / "fairness_checks.csv"
    pd.DataFrame(fairness_rows).to_csv(fairness_path, index=False)

    print(f"Saved raw: {raw_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved fairness: {fairness_path}")
    print(f"Saved commands: {out_root / 'commands.txt'}")


if __name__ == "__main__":
    main()
