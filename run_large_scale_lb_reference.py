#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd


PHI_MIN = 1.0
DELTA_MIN = 0.01


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> List[int]:
    return [int(item) for item in parse_csv(value)]


def safe_tag(value: float | int) -> str:
    if isinstance(value, int) or abs(float(value) - int(float(value))) < 1e-12:
        return str(int(float(value)))
    return f"{float(value):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def parse_points(value: str) -> Dict[str, List[float]]:
    points: Dict[str, List[float]] = {}
    for block in [item.strip() for item in value.split(";") if item.strip()]:
        if ":" not in block:
            raise ValueError(f"Invalid --points block: {block!r}")
        name, raw_values = block.split(":", 1)
        key = name.strip()
        if key not in {"load", "length", "reliability"}:
            raise ValueError(f"Unknown experiment in --points: {key}")
        points[key] = [float(item.strip()) for item in raw_values.split(",") if item.strip()]
    return points


def build_experiment_points(args: argparse.Namespace) -> Dict[str, List[Dict]]:
    raw_points = parse_points(args.points)
    out: Dict[str, List[Dict]] = {}
    out["load"] = [
        {
            "scenario": f"Load $\\lambda={safe_tag(rate)}$",
            "x_name": "arrival_rate",
            "x_value": float(rate),
            "arrival_rate": float(rate),
        }
        for rate in raw_points.get("load", [])
    ]
    out["length"] = [
        {
            "scenario": f"SFC length ${int(sfc_len)}$",
            "x_name": "fixed_sfc_len",
            "x_value": int(sfc_len),
            "arrival_rate": float(args.length_arrival_rate),
            "fixed_sfc_len": int(sfc_len),
        }
        for sfc_len in raw_points.get("length", [])
    ]
    out["reliability"] = [
        {
            "scenario": f"Reliability ${float(reli):.3f}$",
            "x_name": "fixed_reliability",
            "x_value": round(float(reli), 3),
            "arrival_rate": float(args.reliability_arrival_rate),
            "fixed_reliability": round(float(reli), 3),
        }
        for reli in raw_points.get("reliability", [])
    ]
    return out


def build_base_command(args: argparse.Namespace, sim_script: Path) -> List[str]:
    if args.use_current_python:
        return [sys.executable, str(sim_script)]
    return ["conda", "run", "-n", args.conda_env, "python", str(sim_script)]


def method_configs(methods: List[str]) -> List[Dict]:
    configs = {
        "MDCA": {
            "method": "MDCA",
            "algo": "mdca",
            "extra_args": [
                "--max_backup_level",
                "2",
                "--backhaul_bw_scale",
                "0.5",
            ],
        },
        "DMDCA": {
            "method": "DMDCA",
            "algo": "mdca",
            "extra_args": [
                "--max_backup_level",
                "2",
                "--diversity",
                "--p_split",
                "0.3",
                "--explicit_split_replicas",
                "--split_replica_count",
                "2",
                "--split_reliability_mode",
                "identical_active_pool",
                "--backhaul_knn_k",
                "4",
                "--backhaul_bw_scale",
                "0.5",
            ],
        },
    }
    invalid = sorted(set(methods) - set(configs))
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}")
    return [configs[method] for method in methods]


def metric_path(out_dir: Path, region: str, algo: str, seed: int) -> Path:
    return out_dir / f"metrics_{region}_{algo}_seed{seed}.json"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_tasks(args: argparse.Namespace, sim_script: Path, edge_csv: Path, out_root: Path) -> List[Dict]:
    regions = parse_csv(args.regions)
    seeds = parse_int_csv(args.seeds)
    selected_experiments = parse_csv(args.experiments)
    points_by_experiment = build_experiment_points(args)
    invalid_experiments = sorted(set(selected_experiments) - set(points_by_experiment))
    if invalid_experiments:
        raise ValueError(f"Unknown experiments: {invalid_experiments}")

    base_cmd = build_base_command(args, sim_script)
    tasks: List[Dict] = []
    for experiment in selected_experiments:
        points = points_by_experiment.get(experiment, [])
        if not points:
            raise ValueError(f"No points configured for experiment={experiment}")
        for region in regions:
            for point in points:
                point_tag = f"{point['x_name']}_{safe_tag(point['x_value'])}"
                for method_cfg in method_configs(parse_csv(args.methods)):
                    method = method_cfg["method"]
                    algo = method_cfg["algo"]
                    for seed in seeds:
                        run_out_dir = out_root / "split_enabled" / experiment / region / point_tag / method / f"seed{seed}"
                        run_out_dir.mkdir(parents=True, exist_ok=True)
                        cmd = [
                            *base_cmd,
                            "--edge_csv",
                            str(edge_csv),
                            "--region",
                            region,
                            "--time_slots",
                            str(args.time_slots),
                            "--arrival_rate",
                            str(point["arrival_rate"]),
                            "--mean_service_time",
                            str(args.mean_service_time),
                            "--k",
                            str(args.k),
                            "--seed",
                            str(seed),
                            "--algo",
                            algo,
                            "--cap_scale",
                            str(args.cap_scale),
                            "--out_dir",
                            str(run_out_dir),
                            "--record_placement_trace",
                        ]
                        if "fixed_sfc_len" in point:
                            cmd += ["--fixed_sfc_len", str(point["fixed_sfc_len"])]
                        if "fixed_reliability" in point:
                            cmd += ["--fixed_reliability", f"{point['fixed_reliability']:.3f}"]
                        cmd += list(method_cfg["extra_args"])
                        tasks.append(
                            {
                                "section": "split_enabled",
                                "experiment": experiment,
                                "scenario": point["scenario"],
                                "region": region,
                                "x_name": point["x_name"],
                                "x_value": point["x_value"],
                                "method": method,
                                "algo": algo,
                                "seed": seed,
                                "out_dir": run_out_dir,
                                "metric_path": metric_path(run_out_dir, region, algo, seed),
                                "cmd": cmd,
                            }
                        )
    return tasks


def resolve_trace_path(metrics: Dict, metric_file: Path) -> Path:
    raw_path = metrics.get("placement_trace_path")
    if not raw_path:
        raise RuntimeError(f"Missing placement_trace_path in {metric_file}")
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    candidate = Path.cwd() / path
    if candidate.exists():
        return candidate
    return metric_file.parent / path


def compute_trace_lb(trace_path: Path) -> Dict:
    with trace_path.open("r", encoding="utf-8") as handle:
        trace = json.load(handle)
    accepted_trace_count = len(trace)
    instance_count = 0
    split_instance_count = 0
    resource_sum = 0.0
    for request in trace:
        for inst in request.get("instances", []):
            instance_count += 1
            resource_sum += float(inst.get("resource", 0.0))
            if int(inst.get("replica", -1)) >= 0:
                split_instance_count += 1
    lb_total = PHI_MIN * float(instance_count) + DELTA_MIN * float(resource_sum)
    return {
        "accepted_trace_count": accepted_trace_count,
        "instance_count": instance_count,
        "split_instance_count": split_instance_count,
        "resource_sum": resource_sum,
        "lb_trace_total": lb_total,
    }


def analyze_run(task: Dict, metrics: Dict) -> Dict:
    metric_file = Path(task["metric_path"])
    trace_path = resolve_trace_path(metrics, metric_file)
    trace_stats = compute_trace_lb(trace_path)
    total_accepted = int(metrics.get("total_accepted", 0))
    total_cost = float(metrics.get("total_cost", 0.0))
    if trace_stats["accepted_trace_count"] != total_accepted:
        raise RuntimeError(
            f"Trace count mismatch for {metric_file}: "
            f"trace={trace_stats['accepted_trace_count']} metrics={total_accepted}"
        )
    lb_total = float(trace_stats["lb_trace_total"])
    if total_accepted > 0 and lb_total > total_cost + 1e-9:
        raise RuntimeError(f"Lower-bound violation for {metric_file}: LB={lb_total} cost={total_cost}")

    row = {
        "section": task["section"],
        "experiment": task["experiment"],
        "scenario": task["scenario"],
        "region": task["region"],
        "x_name": task["x_name"],
        "x_value": task["x_value"],
        "method": task["method"],
        "algo": task["algo"],
        "seed": task["seed"],
        "trace_hash": metrics.get("trace_hash"),
        "total_arrivals": int(metrics.get("total_arrivals", 0)),
        "total_accepted": total_accepted,
        "acceptance_ratio": float(metrics.get("acceptance_ratio", 0.0)),
        "total_cost": total_cost,
        "avg_cost_per_accepted": float(metrics.get("avg_cost_per_accepted", 0.0)),
        "placement_trace_path": str(trace_path),
        **trace_stats,
    }
    row["lb_trace_per_accepted"] = lb_total / total_accepted if total_accepted > 0 else 0.0
    row["observed_to_lb"] = total_cost / lb_total if lb_total > 0 else 0.0
    row["lb_valid"] = bool(lb_total <= total_cost + 1e-9)
    return row


def run_one_task(task: Dict, skip_existing: bool, dry_run: bool) -> Dict:
    metric_file = Path(task["metric_path"])
    if skip_existing and metric_file.exists():
        metrics = load_json(metric_file)
        row = analyze_run(task, metrics)
        return {"task": task, "row": row, "status": "cached"}

    if dry_run:
        return {"task": task, "row": None, "status": "dry_run", "command": " ".join(task["cmd"])}

    started = time.time()
    completed = subprocess.run(task["cmd"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}) after {elapsed:.1f}s: {' '.join(task['cmd'])}\n\n{completed.stdout}"
        )
    metrics = load_json(metric_file)
    row = analyze_run(task, metrics)
    return {"task": task, "row": row, "status": "ran", "elapsed_s": elapsed}


def ci95(series: pd.Series) -> float:
    n = int(series.count())
    if n <= 1:
        return 0.0
    return float(1.96 * series.std(ddof=1) / (n ** 0.5))


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    grouped_rows: List[Dict] = []
    group_cols = ["experiment", "scenario", "x_name", "x_value", "method", "algo"]
    for key, group in raw_df.groupby(group_cols, sort=False):
        experiment, scenario, x_name, x_value, method, algo = key
        accepted_sum = int(group["total_accepted"].sum())
        cost_sum = float(group["total_cost"].sum())
        lb_sum = float(group["lb_trace_total"].sum())
        row = {
            "experiment": experiment,
            "scenario": scenario,
            "x_name": x_name,
            "x_value": x_value,
            "method": method,
            "algo": algo,
            "runs": int(len(group)),
            "regions": ",".join(sorted(group["region"].unique())),
            "accepted_requests": accepted_sum,
            "observed_cost_total": cost_sum,
            "lb_trace_total": lb_sum,
            "observed_cost_per_accepted": cost_sum / accepted_sum if accepted_sum > 0 else 0.0,
            "lb_trace_per_accepted": lb_sum / accepted_sum if accepted_sum > 0 else 0.0,
            "observed_to_lb": cost_sum / lb_sum if lb_sum > 0 else 0.0,
            "observed_to_lb_mean": float(group["observed_to_lb"].mean()),
            "observed_to_lb_ci95": ci95(group["observed_to_lb"]),
            "accepted_per_run_mean": float(group["total_accepted"].mean()),
            "accepted_per_run_ci95": ci95(group["total_accepted"]),
            "avg_cost_per_accepted_mean": float(group["avg_cost_per_accepted"].mean()),
            "avg_cost_per_accepted_ci95": ci95(group["avg_cost_per_accepted"]),
            "avg_lb_per_accepted_mean": float(group["lb_trace_per_accepted"].mean()),
            "avg_lb_per_accepted_ci95": ci95(group["lb_trace_per_accepted"]),
            "split_instance_count": int(group["split_instance_count"].sum()),
            "instance_count": int(group["instance_count"].sum()),
        }
        grouped_rows.append(row)
    return pd.DataFrame(grouped_rows)


def write_latex_table(summary_df: pd.DataFrame, out_path: Path) -> None:
    selected = summary_df[
        (
            (summary_df["experiment"] == "load")
            & (summary_df["x_value"].astype(float).round(6) == 50.0)
        )
        | (
            (summary_df["experiment"] == "length")
            & (summary_df["x_value"].astype(float).round(6) == 6.0)
        )
        | (
            (summary_df["experiment"] == "reliability")
            & (summary_df["x_value"].astype(float).round(6) == 0.999)
        )
    ].copy()
    experiment_order = {"load": 0, "length": 1, "reliability": 2}
    method_order = {"MDCA": 0, "DMDCA": 1}
    selected["_experiment_order"] = selected["experiment"].map(experiment_order)
    selected["_method_order"] = selected["method"].map(method_order)
    selected = selected.sort_values(["_experiment_order", "x_value", "_method_order"])

    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Scenario & Method & Runs & Accepted & Cost/accepted & LB/accepted & Cost/LB \\\\",
        "\\midrule",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"{row['scenario']} & {row['method']} & {int(row['runs'])} & {int(row['accepted_requests'])} & "
            f"{float(row['observed_cost_per_accepted']):.2f} & {float(row['lb_trace_per_accepted']):.2f} & "
            f"{float(row['observed_to_lb']):.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def validate_fairness(raw_df: pd.DataFrame, out_root: Path) -> None:
    fairness_cols = ["experiment", "region", "x_name", "x_value", "seed"]
    fairness_df = raw_df.groupby(fairness_cols).agg(
        trace_hash_nunique=("trace_hash", "nunique"),
        total_arrivals_nunique=("total_arrivals", "nunique"),
    ).reset_index()
    fairness_df["trace_consistent"] = (
        (fairness_df["trace_hash_nunique"] == 1)
        & (fairness_df["total_arrivals_nunique"] == 1)
    )
    fairness_path = out_root / "fairness_checks.csv"
    fairness_df.to_csv(fairness_path, index=False)
    if not fairness_df["trace_consistent"].all():
        bad = fairness_df.loc[~fairness_df["trace_consistent"]]
        raise RuntimeError(f"Trace consistency check failed:\n{bad.to_string(index=False)}")


def save_outputs(raw_rows: List[Dict], out_root: Path) -> None:
    raw_df = pd.DataFrame(raw_rows)
    if raw_df.empty:
        raise RuntimeError("No rows were produced.")
    if not raw_df["lb_valid"].all():
        bad = raw_df.loc[~raw_df["lb_valid"]]
        raise RuntimeError(f"Lower-bound validation failed:\n{bad.to_string(index=False)}")

    raw_path = out_root / "raw.csv"
    raw_df.to_csv(raw_path, index=False)
    validate_fairness(raw_df, out_root)

    summary_df = summarize(raw_df)
    summary_path = out_root / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    table_path = out_root / "large_scale_lb_reference_table.tex"
    write_latex_table(summary_df, table_path)

    print(f"Saved raw rows: {raw_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved LaTeX table: {table_path}")
    print(f"Validated {len(raw_df)} runs with LB_trace <= observed total cost.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_script", type=str, default="online_mdca_sim_delay_split_strict.py")
    parser.add_argument("--conda_env", type=str, default="SFC")
    parser.add_argument("--use_current_python", action="store_true")
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--edge_csv", type=str, default="Edge_devices.csv")
    parser.add_argument("--regions", type=str, default="Alabama,Arizona")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--experiments", type=str, default="load,length,reliability")
    parser.add_argument("--points", type=str, default="load:20,35,50;length:1,3,6;reliability:0.990,0.996,0.999")
    parser.add_argument("--methods", type=str, default="MDCA,DMDCA")
    parser.add_argument("--time_slots", type=int, default=120)
    parser.add_argument("--mean_service_time", type=float, default=9.0)
    parser.add_argument("--cap_scale", type=float, default=0.2)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--length_arrival_rate", type=float, default=20.0)
    parser.add_argument("--reliability_arrival_rate", type=float, default=12.0)
    parser.add_argument("--out_dir", type=str, default="paper_data/large_scale_lb_reference")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--table_only",
        action="store_true",
        help="Regenerate the LaTeX table from an existing summary.csv without running simulations.",
    )
    args = parser.parse_args()

    sim_script = Path(args.sim_script).resolve()
    edge_csv = Path(args.edge_csv).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.table_only:
        summary_path = out_root / "summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary file for --table_only: {summary_path}")
        summary_df = pd.read_csv(summary_path)
        table_path = out_root / "large_scale_lb_reference_table.tex"
        write_latex_table(summary_df, table_path)
        print(f"Regenerated LaTeX table: {table_path}")
        return

    tasks = build_tasks(args, sim_script, edge_csv, out_root)
    print(
        f"Prepared {len(tasks)} runs | regions={args.regions} | seeds={args.seeds} | "
        f"experiments={args.experiments} | methods={args.methods} | out_dir={out_root}",
        flush=True,
    )

    if args.dry_run:
        for task in tasks:
            result = run_one_task(task, args.skip_existing, dry_run=True)
            print(result["command"])
        print("Dry run complete.")
        return

    raw_rows: List[Dict] = []
    completed_count = 0
    started_all = time.time()

    def append_result(result: Dict) -> None:
        nonlocal completed_count
        task = result["task"]
        row = result["row"]
        if row is not None:
            raw_rows.append(row)
        completed_count += 1
        elapsed = time.time() - started_all
        rate = elapsed / max(completed_count, 1)
        eta = rate * (len(tasks) - completed_count)
        if completed_count == len(tasks) or completed_count % max(args.progress_every, 1) == 0:
            print(
                f"[{completed_count}/{len(tasks)}] {task['experiment']}/{task['region']}/{task['method']}/seed{task['seed']} "
                f"status={result['status']} run_s={result.get('elapsed_s', 0.0):.1f} eta_min={eta / 60.0:.1f}",
                flush=True,
            )

    if args.max_workers <= 1:
        for task in tasks:
            append_result(run_one_task(task, args.skip_existing, dry_run=False))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {
                executor.submit(run_one_task, task, args.skip_existing, False): task
                for task in tasks
            }
            for future in concurrent.futures.as_completed(future_map):
                append_result(future.result())

    save_outputs(raw_rows, out_root)


if __name__ == "__main__":
    main()
