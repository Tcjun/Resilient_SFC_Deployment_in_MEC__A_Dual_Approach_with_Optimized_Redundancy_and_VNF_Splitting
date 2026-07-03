#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def parse_int_csv(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_csv(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def safe_tag(value: float | int | str) -> str:
    if isinstance(value, str):
        return value.replace(".", "p")
    if abs(float(value) - int(float(value))) < 1e-12:
        return str(int(float(value)))
    return f"{float(value):g}".replace(".", "p")


def metric_path(out_dir: Path, region: str, seed: int) -> Path:
    return out_dir / f"metrics_{region}_mdca_seed{seed}.json"


def runtime_path(out_dir: Path, region: str, seed: int) -> Path:
    return out_dir / f"runtime_{region}_mdca_seed{seed}.json"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def ci95(series: pd.Series) -> float:
    n = int(series.count())
    if n <= 1:
        return 0.0
    return float(1.96 * series.std(ddof=1) / math.sqrt(n))


def command_to_text(cmd: Iterable[str]) -> str:
    return " ".join(str(part) for part in cmd)


def create_nested_region_subsets(edge_csv: Path, out_root: Path, region: str, k_values: List[int], subset_seed: int) -> Dict[int, Path]:
    df = pd.read_csv(edge_csv)
    if "Region" not in df.columns:
        raise ValueError(f"{edge_csv} does not contain a Region column.")
    region_df = df[df["Region"] == region].copy()
    if region_df.empty:
        raise ValueError(f"No rows found for region={region} in {edge_csv}.")
    max_k = max(k_values)
    if max_k > len(region_df):
        raise ValueError(f"Requested K={max_k}, but region={region} only has {len(region_df)} cloudlets.")

    rng = np.random.default_rng(int(subset_seed))
    permutation = rng.permutation(len(region_df))
    subset_dir = out_root / "subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[int, Path] = {}
    manifest_rows: List[Dict] = []
    for k_value in sorted(k_values):
        selected_positions = sorted(int(pos) for pos in permutation[: int(k_value)])
        subset_df = region_df.iloc[selected_positions].copy()
        subset_path = subset_dir / f"Edge_devices_{region}_K{int(k_value):03d}_subset_seed{subset_seed}.csv"
        subset_df.to_csv(subset_path, index=False)
        paths[int(k_value)] = subset_path
        for rank, original_position in enumerate(selected_positions):
            manifest_rows.append(
                {
                    "region": region,
                    "K": int(k_value),
                    "subset_seed": int(subset_seed),
                    "subset_rank": int(rank),
                    "region_row_position": int(original_position),
                }
            )

    pd.DataFrame(manifest_rows).to_csv(subset_dir / "subset_manifest.csv", index=False)
    return paths


def method_extra_args(method: str, args: argparse.Namespace) -> List[str]:
    base = ["--max_backup_level", str(args.max_backup_level)]
    if method == "MDCA":
        return base
    if method == "DMDCA":
        return [
            *base,
            "--diversity",
            "--p_split",
            str(args.p_split),
            "--explicit_split_replicas",
            "--split_replica_count",
            str(args.split_replica_count),
            "--split_reliability_mode",
            "identical_active_pool",
        ]
    raise ValueError(f"Unknown method={method}")


def build_sim_command(
    python_executable: str,
    sim_script: Path,
    edge_csv: Path,
    region: str,
    seed: int,
    out_dir: Path,
    arrival_rate: float,
    fixed_sfc_len: int,
    method: str,
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        python_executable,
        str(sim_script),
        "--edge_csv",
        str(edge_csv),
        "--region",
        region,
        "--time_slots",
        str(args.time_slots),
        "--arrival_rate",
        str(arrival_rate),
        "--mean_service_time",
        str(args.mean_service_time),
        "--k",
        str(args.k),
        "--algo",
        "mdca",
        "--cap_scale",
        str(args.cap_scale),
        "--backhaul_knn_k",
        str(args.backhaul_knn_k),
        "--backhaul_bw_scale",
        str(args.backhaul_bw_scale),
        "--fixed_reliability",
        f"{float(args.fixed_reliability):.6g}",
        "--fixed_sfc_len",
        str(fixed_sfc_len),
        "--seed",
        str(seed),
        "--out_dir",
        str(out_dir),
    ]
    cmd.extend(method_extra_args(method, args))
    return cmd


def build_tasks(args: argparse.Namespace, sim_script: Path, edge_csv: Path, out_root: Path) -> List[Dict]:
    seeds = parse_int_csv(args.seeds)
    k_values = parse_int_csv(args.k_values)
    load_rates = parse_float_csv(args.load_rates)
    length_values = parse_int_csv(args.length_values)
    methods = ["MDCA", "DMDCA"]
    tasks: List[Dict] = []

    subset_paths = create_nested_region_subsets(
        edge_csv=edge_csv,
        out_root=out_root,
        region=args.k_sweep_region,
        k_values=k_values,
        subset_seed=args.subset_seed,
    )

    for k_value in k_values:
        for method in methods:
            for seed in seeds:
                region = args.k_sweep_region
                x_name = "cloudlet_count"
                x_value = int(k_value)
                out_dir = out_root / "runs" / "k_sweep" / region / f"K_{safe_tag(k_value)}" / method / f"seed{seed}"
                cmd = build_sim_command(
                    python_executable=args.python_executable,
                    sim_script=sim_script,
                    edge_csv=subset_paths[int(k_value)],
                    region=region,
                    seed=seed,
                    out_dir=out_dir,
                    arrival_rate=float(args.k_sweep_arrival_rate),
                    fixed_sfc_len=int(args.k_sweep_sfc_len),
                    method=method,
                    args=args,
                )
                tasks.append(
                    {
                        "experiment": "k_sweep",
                        "region": region,
                        "x_name": x_name,
                        "x_value": x_value,
                        "method": method,
                        "seed": seed,
                        "edge_csv": subset_paths[int(k_value)],
                        "out_dir": out_dir,
                        "cmd": cmd,
                    }
                )

    for region in ["Alabama", "Arizona"]:
        for load_rate in load_rates:
            for method in methods:
                for seed in seeds:
                    x_name = "arrival_rate"
                    x_value = float(load_rate)
                    out_dir = out_root / "runs" / "load_sweep" / region / f"arrival_rate_{safe_tag(load_rate)}" / method / f"seed{seed}"
                    cmd = build_sim_command(
                        python_executable=args.python_executable,
                        sim_script=sim_script,
                        edge_csv=edge_csv,
                        region=region,
                        seed=seed,
                        out_dir=out_dir,
                        arrival_rate=float(load_rate),
                        fixed_sfc_len=int(args.load_sweep_sfc_len),
                        method=method,
                        args=args,
                    )
                    tasks.append(
                        {
                            "experiment": "load_sweep",
                            "region": region,
                            "x_name": x_name,
                            "x_value": x_value,
                            "method": method,
                            "seed": seed,
                            "edge_csv": edge_csv,
                            "out_dir": out_dir,
                            "cmd": cmd,
                        }
                    )

    for length_value in length_values:
        for method in methods:
            for seed in seeds:
                region = args.length_sweep_region
                x_name = "fixed_sfc_len"
                x_value = int(length_value)
                out_dir = out_root / "runs" / "length_sweep" / region / f"fixed_sfc_len_{safe_tag(length_value)}" / method / f"seed{seed}"
                cmd = build_sim_command(
                    python_executable=args.python_executable,
                    sim_script=sim_script,
                    edge_csv=edge_csv,
                    region=region,
                    seed=seed,
                    out_dir=out_dir,
                    arrival_rate=float(args.length_sweep_arrival_rate),
                    fixed_sfc_len=int(length_value),
                    method=method,
                    args=args,
                )
                tasks.append(
                    {
                        "experiment": "length_sweep",
                        "region": region,
                        "x_name": x_name,
                        "x_value": x_value,
                        "method": method,
                        "seed": seed,
                        "edge_csv": edge_csv,
                        "out_dir": out_dir,
                        "cmd": cmd,
                    }
                )

    return tasks


def run_one_task(task: Dict, force: bool, dry_run: bool) -> Dict:
    out_dir = Path(task["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    region = str(task["region"])
    seed = int(task["seed"])
    metrics_file = metric_path(out_dir, region, seed)
    runtime_file = runtime_path(out_dir, region, seed)

    if dry_run:
        return {"task": task, "status": "dry_run", "command": command_to_text(task["cmd"])}

    if not force and metrics_file.exists() and runtime_file.exists():
        metrics = load_json(metrics_file)
        runtime = load_json(runtime_file)
        return {"task": task, "metrics": metrics, "runtime": runtime, "status": "cached"}

    started = time.perf_counter()
    completed = subprocess.run(task["cmd"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed "
            f"({completed.returncode}) after {elapsed:.1f}s:\n{command_to_text(task['cmd'])}\n\n{completed.stdout}"
        )
    metrics = load_json(metrics_file)
    runtime = {
        "total_runtime_s": float(elapsed),
        "command": command_to_text(task["cmd"]),
        "completed_returncode": int(completed.returncode),
    }
    write_json(runtime_file, runtime)
    return {"task": task, "metrics": metrics, "runtime": runtime, "status": "ran"}


def arrivals_key(value: object) -> str:
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def summarize(raw_rows: List[Dict], out_root: Path) -> None:
    raw_df = pd.DataFrame(raw_rows)
    raw_path = out_root / "raw_metrics_long.csv"
    raw_df.to_csv(raw_path, index=False)

    fairness_rows: List[Dict] = []
    fairness_cols = ["experiment", "region", "x_name", "x_value", "seed"]
    for keys, group in raw_df.groupby(fairness_cols):
        arrival_keys = group["arrivals_per_slot"].apply(arrivals_key)
        fairness_rows.append(
            {
                "experiment": keys[0],
                "region": keys[1],
                "x_name": keys[2],
                "x_value": keys[3],
                "seed": keys[4],
                "method_count": int(group["method"].nunique()),
                "trace_hash_nunique": int(group["trace_hash"].nunique()),
                "arrivals_per_slot_nunique": int(arrival_keys.nunique()),
                "total_arrivals_nunique": int(group["total_arrivals"].nunique()),
                "trace_consistent": bool(
                    group["method"].nunique() == 2
                    and group["trace_hash"].nunique() == 1
                    and arrival_keys.nunique() == 1
                    and group["total_arrivals"].nunique() == 1
                ),
            }
        )
    fairness = pd.DataFrame(fairness_rows)
    fairness_path = out_root / "fairness_checks.csv"
    fairness.to_csv(fairness_path, index=False)
    if not fairness["trace_consistent"].all():
        bad = fairness.loc[~fairness["trace_consistent"]]
        raise RuntimeError(f"Trace consistency check failed:\n{bad.to_string(index=False)}")

    metric_cols = [
        "total_runtime_s",
        "avg_runtime_per_slot_s",
        "avg_runtime_per_arrival_ms",
        "acceptance_ratio",
        "avg_cost_per_accepted",
        "avg_e2e_delay_accepted",
        "p95_e2e_delay_accepted",
        "total_arrivals",
        "total_accepted",
        "region_cloudlet_count",
    ]
    group_cols = ["experiment", "region", "x_name", "x_value", "method"]
    summary = raw_df.groupby(group_cols)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns]
    for metric in metric_cols:
        ci = raw_df.groupby(group_cols)[metric].apply(ci95).reset_index(name=f"{metric}_ci95")
        summary = summary.merge(ci, on=group_cols, how="left")
    summary_path = out_root / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved raw metrics: {raw_path}")
    print(f"Saved fairness checks: {fairness_path}")
    print(f"Saved summary metrics: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_script", default="online_mdca_sim_delay_split_strict.py")
    parser.add_argument("--edge_csv", default="Edge_devices.csv")
    parser.add_argument("--out_root", default="paper_data/scalability_runtime")
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--time_slots", type=int, default=120)
    parser.add_argument("--mean_service_time", type=float, default=9.0)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--cap_scale", type=float, default=0.2)
    parser.add_argument("--backhaul_knn_k", type=int, default=4)
    parser.add_argument("--backhaul_bw_scale", type=float, default=0.5)
    parser.add_argument("--fixed_reliability", type=float, default=0.999)
    parser.add_argument("--max_backup_level", type=int, default=2)
    parser.add_argument("--p_split", type=float, default=0.3)
    parser.add_argument("--split_replica_count", type=int, default=2)
    parser.add_argument("--subset_seed", type=int, default=20260610)
    parser.add_argument("--k_sweep_region", default="Arizona")
    parser.add_argument("--k_values", default="50,100,200,314")
    parser.add_argument("--k_sweep_arrival_rate", type=float, default=20.0)
    parser.add_argument("--k_sweep_sfc_len", type=int, default=3)
    parser.add_argument("--load_rates", default="10,30,50")
    parser.add_argument("--load_sweep_sfc_len", type=int, default=3)
    parser.add_argument("--length_sweep_region", default="Arizona")
    parser.add_argument("--length_values", default="1,3,6")
    parser.add_argument("--length_sweep_arrival_rate", type=float, default=20.0)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    sim_script = Path(args.sim_script).resolve()
    edge_csv = Path(args.edge_csv).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args, sim_script, edge_csv, out_root)
    commands_path = out_root / "commands.txt"
    commands_path.write_text("\n".join(command_to_text(task["cmd"]) for task in tasks) + "\n", encoding="utf-8")
    config = {
        "task_count": len(tasks),
        "python_executable": args.python_executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "edge_csv": str(edge_csv),
        "sim_script": str(sim_script),
        "out_root": str(out_root),
        "runtime_note": "Wall-clock runtime measured by the scalability runner around each simulator process.",
        "interpretation_note": "K sweep changes topology and capacity; runtime is the primary scalability metric.",
        "args": vars(args),
    }
    write_json(out_root / "run_config.json", config)

    print(
        f"Prepared {len(tasks)} runs | out_root={out_root} | dry_run={args.dry_run} | "
        f"max_workers={args.max_workers}",
        flush=True,
    )
    if args.dry_run:
        print(f"Saved commands: {commands_path}")
        return

    raw_rows: List[Dict] = []
    started_all = time.perf_counter()
    completed_count = 0

    def append_result(result: Dict) -> None:
        nonlocal completed_count
        task = result["task"]
        metrics = dict(result["metrics"])
        runtime = dict(result["runtime"])
        total_runtime_s = float(runtime["total_runtime_s"])
        total_arrivals = int(metrics.get("total_arrivals", 0))
        time_slots = int(metrics.get("time_slots", args.time_slots))
        row = {
            "experiment": task["experiment"],
            "region": task["region"],
            "x_name": task["x_name"],
            "x_value": task["x_value"],
            "method": task["method"],
            "seed": int(task["seed"]),
            "edge_csv_used": str(task["edge_csv"]),
            "out_dir": str(task["out_dir"]),
            "status": result["status"],
            "command": command_to_text(task["cmd"]),
            "total_runtime_s": total_runtime_s,
            "avg_runtime_per_slot_s": total_runtime_s / max(time_slots, 1),
            "avg_runtime_per_arrival_ms": (1000.0 * total_runtime_s / total_arrivals) if total_arrivals > 0 else 0.0,
        }
        row.update(metrics)
        raw_rows.append(row)
        completed_count += 1
        elapsed = time.perf_counter() - started_all
        eta = elapsed / max(completed_count, 1) * (len(tasks) - completed_count)
        if completed_count == len(tasks) or completed_count % max(int(args.progress_every), 1) == 0:
            print(
                f"[{completed_count}/{len(tasks)}] {task['experiment']}/{task['region']}/{task['x_name']}={task['x_value']}/"
                f"{task['method']}/seed{task['seed']} status={result['status']} "
                f"run_s={total_runtime_s:.2f} eta_min={eta / 60.0:.1f}",
                flush=True,
            )

    if args.max_workers <= 1:
        for task in tasks:
            append_result(run_one_task(task, force=args.force, dry_run=False))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {executor.submit(run_one_task, task, args.force, False): task for task in tasks}
            for future in concurrent.futures.as_completed(future_map):
                append_result(future.result())

    summarize(raw_rows, out_root)
    print(f"Saved commands: {commands_path}")
    print(f"Saved config: {out_root / 'run_config.json'}")


if __name__ == "__main__":
    main()
