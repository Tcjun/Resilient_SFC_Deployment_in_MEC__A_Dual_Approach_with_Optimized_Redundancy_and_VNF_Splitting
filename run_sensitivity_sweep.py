#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch sensitivity sweep runner for the strict online SFC simulator.

It repeatedly runs:
  online_mdca_sim_delay_split_strict.py

over a grid of splitting-delay parameters (tau_cons, rho_sync, etc.) and seeds,
then aggregates metrics into CSV summary tables (mean/std/95% CI).

Example:
  python run_sensitivity_sweep.py \
    --sim_script online_mdca_sim_delay_split_strict.py \
    --edge_csv Edge_devices.csv --region Alabama \
    --time_slots 120 --arrival_rate 8 --mean_service_time 5 \
    --seeds 1,2,3,4,5 \
    --diversity --p_split 0.3 \
    --tau_cons_ms 0,0.2,0.5,1,2 \
    --rho_sync 0,0.01,0.05,0.1 \
    --tau_coord_ms 0.1 \
    --tau_merge_ms 0.05 \
    --split_replica_count 2 --split_delay_overhead_ratio 0.15 \
    --backhaul_knn_k 4 --backhaul_bw_scale 0.5 \
    --out_root ./sweep_runs
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _parse_csv_floats(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x) for x in s.split(",")]


def _parse_csv_ints(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",")]


def _safe_tag(x: float) -> str:
    # 0.1 -> 0p1, 1.0 -> 1, 0.05 -> 0p05
    if abs(x - int(x)) < 1e-12:
        return str(int(x))
    s = f"{x}".replace(".", "p")
    return s


def run_one(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}):\n{' '.join(cmd)}\n\nOutput:\n{p.stdout}")


def read_metrics(out_dir: Path, region: str, seed: int) -> Dict:
    mpath = out_dir / f"metrics_{region}_seed{seed}.json"
    if not mpath.exists():
        raise FileNotFoundError(f"Missing metrics file: {mpath}")
    with open(mpath, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_script", type=str, default="online_mdca_sim_delay_split_strict.py")
    ap.add_argument("--edge_csv", type=str, default="Edge_devices.csv")
    ap.add_argument("--region", type=str, default="Alabama")
    ap.add_argument("--time_slots", type=int, default=120)
    ap.add_argument("--arrival_rate", type=float, default=8.0)
    ap.add_argument("--mean_service_time", type=float, default=5.0)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seeds", type=str, default="1,2,3,4,5")

    # algorithm toggles
    ap.add_argument("--diversity", action="store_true")
    ap.add_argument("--p_split", type=float, default=0.3)

    # splitting params (grid)
    ap.add_argument("--tau_cons_ms", type=str, default="0,0.2,0.5,1,2")
    ap.add_argument("--rho_sync", type=str, default="0,0.01,0.05,0.1")

    # optional: keep coord/merge fixed or sweep too
    ap.add_argument("--tau_coord_ms", type=str, default="0.1")
    ap.add_argument("--tau_merge_ms", type=str, default="0.05")

    ap.add_argument("--split_replica_count", type=int, default=2)
    ap.add_argument("--split_delay_overhead_ratio", type=float, default=0.15)

    # backhaul strict options
    ap.add_argument("--backhaul_knn_k", type=int, default=4)
    ap.add_argument("--backhaul_bw_scale", type=float, default=0.5)
    ap.add_argument("--fiber_speed_mps", type=float, default=2e8)
    ap.add_argument("--switch_delay_s", type=float, default=2e-4)

    ap.add_argument("--out_root", type=str, default="./sweep_runs")
    ap.add_argument("--dry_run", action="store_true", help="Only print commands without running")

    args = ap.parse_args()

    sim_script = Path(args.sim_script).resolve()
    edge_csv = Path(args.edge_csv).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    seeds = _parse_csv_ints(args.seeds)
    tau_cons_ms_list = _parse_csv_floats(args.tau_cons_ms)
    rho_sync_list = _parse_csv_floats(args.rho_sync)
    tau_coord_ms_list = _parse_csv_floats(args.tau_coord_ms)
    tau_merge_ms_list = _parse_csv_floats(args.tau_merge_ms)

    grid = list(itertools.product(tau_cons_ms_list, rho_sync_list, tau_coord_ms_list, tau_merge_ms_list))
    print(f"Grid size: {len(grid)} combos; seeds: {len(seeds)}; total runs: {len(grid)*len(seeds)}")

    rows = []
    for (tau_cons_ms, rho_sync, tau_coord_ms, tau_merge_ms) in grid:
        # Unique outdir per combo to avoid overwriting metrics_{region}_seedX.json
        combo_tag = (
            f"V{args.split_replica_count}"
            f"_g{_safe_tag(args.split_delay_overhead_ratio)}"
            f"_cons{_safe_tag(tau_cons_ms)}ms"
            f"_sync{_safe_tag(rho_sync)}"
            f"_coord{_safe_tag(tau_coord_ms)}ms"
            f"_merge{_safe_tag(tau_merge_ms)}ms"
            f"_k{args.backhaul_knn_k}"
            f"_bw{_safe_tag(args.backhaul_bw_scale)}"
        )
        out_dir = out_root / args.region / combo_tag
        out_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            cmd = [
                sys.executable, str(sim_script),
                "--edge_csv", str(edge_csv),
                "--region", args.region,
                "--time_slots", str(args.time_slots),
                "--arrival_rate", str(args.arrival_rate),
                "--mean_service_time", str(args.mean_service_time),
                "--k", str(args.k),
                "--seed", str(seed),
                "--out_dir", str(out_dir),
                "--backhaul_knn_k", str(args.backhaul_knn_k),
                "--backhaul_bw_scale", str(args.backhaul_bw_scale),
                "--fiber_speed_mps", str(args.fiber_speed_mps),
                "--switch_delay_s", str(args.switch_delay_s),
                "--split_replica_count", str(args.split_replica_count),
                "--split_delay_overhead_ratio", str(args.split_delay_overhead_ratio),
                "--split_coord_delay_s", str(tau_coord_ms / 1000.0),
                "--split_merge_delay_s", str(tau_merge_ms / 1000.0),
                "--split_consistency_delay_s", str(tau_cons_ms / 1000.0),
                "--split_sync_traffic_ratio", str(rho_sync),
            ]
            if args.diversity:
                cmd += ["--diversity", "--p_split", str(args.p_split)]

            if args.dry_run:
                print(" ".join(cmd))
                continue

            run_one(cmd)
            m = read_metrics(out_dir, args.region, seed)

            # attach sweep params
            m["_tau_cons_ms"] = tau_cons_ms
            m["_rho_sync"] = rho_sync
            m["_tau_coord_ms"] = tau_coord_ms
            m["_tau_merge_ms"] = tau_merge_ms
            m["_split_replica_count"] = args.split_replica_count
            m["_split_delay_overhead_ratio"] = args.split_delay_overhead_ratio
            m["_backhaul_knn_k"] = args.backhaul_knn_k
            m["_backhaul_bw_scale"] = args.backhaul_bw_scale

            rows.append(m)

    if args.dry_run:
        print("Dry run complete.")
        return

    raw_df = pd.DataFrame(rows)

    # Keep a stable set of key metrics (safe even if some are missing)
    metric_cols = [
        "acceptance_ratio",
        "avg_cost_per_accepted",
        "avg_cost_per_arrival",
        "avg_e2e_delay_accepted",
        "p95_e2e_delay_accepted",
        "avg_processing_delay_accepted",
        "avg_network_delay_accepted",
        "avg_split_proc_overhead_accepted",
        "avg_split_net_overhead_accepted",
        "avg_split_overhead_ratio_accepted",
        "p95_split_overhead_ratio_accepted",
        "qos_rejection_ratio",
        "final_utilization",
        "total_split_transmitted_bits_accepted",
        "avg_split_transmitted_bits_per_accepted",
    ]
    keep_cols = ["region", "seed", "diversity", "p_split",
                 "_tau_cons_ms", "_rho_sync", "_tau_coord_ms", "_tau_merge_ms",
                 "_split_replica_count", "_split_delay_overhead_ratio",
                 "_backhaul_knn_k", "_backhaul_bw_scale"] + [c for c in metric_cols if c in raw_df.columns]
    raw_df = raw_df[keep_cols].copy()

    raw_path = out_root / args.region / "raw_metrics.csv"
    raw_df.to_csv(raw_path, index=False)

    # Summary across seeds
    group_cols = ["region", "diversity", "p_split",
                  "_tau_cons_ms", "_rho_sync", "_tau_coord_ms", "_tau_merge_ms",
                  "_split_replica_count", "_split_delay_overhead_ratio",
                  "_backhaul_knn_k", "_backhaul_bw_scale"]

    agg = {}
    for c in metric_cols:
        if c in raw_df.columns:
            agg[c] = ["mean", "std", "count"]

    summ = raw_df.groupby(group_cols).agg(agg)
    # flatten columns
    summ.columns = ["__".join(col).rstrip("_") for col in summ.columns.to_flat_index()]
    summ = summ.reset_index()

    # 95% CI using normal approximation (good enough for seed sweeps)
    for c in metric_cols:
        mean_col = f"{c}__mean"
        std_col = f"{c}__std"
        n_col = f"{c}__count"
        if mean_col in summ.columns and std_col in summ.columns and n_col in summ.columns:
            summ[f"{c}__ci95"] = 1.96 * (summ[std_col].fillna(0.0) / summ[n_col].clip(lower=1).pow(0.5))

    summ_path = out_root / args.region / "summary_metrics.csv"
    summ.to_csv(summ_path, index=False)

    print(f"Saved raw: {raw_path}")
    print(f"Saved summary: {summ_path}")


if __name__ == "__main__":
    main()
