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


def parse_int_csv(value: str) -> List[int]:
    return [int(item) for item in value.split(',') if item.strip()]


def parse_float_csv(value: str) -> List[float]:
    return [float(item) for item in value.split(',') if item.strip()]


def safe_tag(value: float | int) -> str:
    if isinstance(value, int) or abs(float(value) - int(float(value))) < 1e-12:
        return str(int(float(value)))
    return str(value).replace('.', 'p')


def metric_path(out_dir: Path, region: str, algo: str, seed: int) -> Path:
    return out_dir / f"metrics_{region}_{algo}_seed{seed}.json"


def load_metrics(path: Path) -> Dict:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def build_experiment_points(args: argparse.Namespace) -> Dict[str, List[Dict]]:
    load_points = [{"x_name": "arrival_rate", "x_value": rate, "arrival_rate": rate} for rate in parse_float_csv(args.load_rates)]
    length_points = [
        {"x_name": "fixed_sfc_len", "x_value": int(sfc_len), "arrival_rate": float(args.length_arrival_rate), "fixed_sfc_len": int(sfc_len)}
        for sfc_len in parse_int_csv(args.sfc_lengths)
    ]
    reli_points = [
        {
            "x_name": "fixed_reliability",
            "x_value": round(reli, 3),
            "arrival_rate": float(args.reliability_arrival_rate),
            "fixed_reliability": round(reli, 3),
        }
        for reli in parse_float_csv(args.reliability_points)
    ]
    return {"load": load_points, "length": length_points, "reliability": reli_points}


def build_base_command(args: argparse.Namespace, sim_script: Path) -> List[str]:
    if args.use_current_python:
        return [sys.executable, str(sim_script)]
    return ['conda', 'run', '-n', args.conda_env, 'python', str(sim_script)]


def build_tasks(args: argparse.Namespace, sim_script: Path, edge_csv: Path, out_root: Path) -> List[Dict]:
    regions = [item.strip() for item in args.regions.split(',') if item.strip()]
    seeds = parse_int_csv(args.seeds)
    selected_experiments = [item.strip() for item in args.experiments.split(',') if item.strip()]
    experiment_points_all = build_experiment_points(args)
    invalid_experiments = sorted(set(selected_experiments) - set(experiment_points_all))
    if invalid_experiments:
        raise ValueError(f'Unknown experiments: {invalid_experiments}')
    experiment_points = {
        name: (experiment_points_all[name][: args.max_points_per_experiment] if args.max_points_per_experiment else experiment_points_all[name])
        for name in selected_experiments
    }

    pvfp_extra_args = ['--max_backup_level', '2']
    if args.pvfp_model_path:
        pvfp_extra_args += ['--pvfp_model_path', str(Path(args.pvfp_model_path).resolve())]
    if args.pvfp_disable_fallback:
        pvfp_extra_args += ['--pvfp_disable_fallback']

    mdca_methods = [
        {"method": "MDCA", "algo": "mdca", "extra_args": ['--max_backup_level', '2']},
        {"method": "DTSP", "algo": "dtsp", "extra_args": ['--max_backup_level', '2']},
        {"method": "HSPA", "algo": "hspa", "extra_args": ['--max_backup_level', '2']},
        {"method": "Random", "algo": "random", "extra_args": ['--max_backup_level', '2']},
        {"method": "PVFP-FDRL", "algo": "pvfp_fdrl", "extra_args": list(pvfp_extra_args)},
    ]
    dmdca_methods = [
        {"method": "MDCA", "algo": "mdca", "extra_args": ['--max_backup_level', '2', '--backhaul_bw_scale', '0.5']},
        {
            "method": "DMDCA",
            "algo": "mdca",
            "extra_args": [
                '--max_backup_level', '2',
                '--diversity',
                '--p_split', '0.3',
                '--explicit_split_replicas',
                '--split_replica_count', '2',
                '--split_reliability_mode', 'identical_active_pool',
                '--backhaul_knn_k', '4', '--backhaul_bw_scale', '0.5',
            ],
        },
        {
            "method": "PVFP-FDRL",
            "algo": "pvfp_fdrl",
            "extra_args": [
                *pvfp_extra_args,
                '--diversity',
                '--p_split', '0.3',
                '--explicit_split_replicas',
                '--split_replica_count', '2',
                '--split_reliability_mode', 'identical_active_pool',
                '--backhaul_knn_k', '4', '--backhaul_bw_scale', '0.5',
            ],
        },
    ]

    section_aliases = {
        'mdca': 'unsplit',
        'dmdca': 'split_enabled',
        'unsplit': 'unsplit',
        'split_enabled': 'split_enabled',
    }
    selected_sections = ['unsplit', 'split_enabled'] if args.section == 'all' else [section_aliases[args.section]]
    base_cmd = build_base_command(args, sim_script)
    tasks: List[Dict] = []

    for section in selected_sections:
        methods = mdca_methods if section == 'unsplit' else dmdca_methods
        for experiment in selected_experiments:
            points = experiment_points[experiment]
            for region in regions:
                for point in points:
                    x_name = point['x_name']
                    x_value = point['x_value']
                    point_tag = f"{x_name}_{safe_tag(x_value)}"
                    for method_cfg in methods:
                        method = method_cfg['method']
                        algo = method_cfg['algo']
                        for seed in seeds:
                            out_dir = out_root / section / experiment / region / point_tag / method / f"seed{seed}"
                            out_dir.mkdir(parents=True, exist_ok=True)
                            mpath = metric_path(out_dir, region, algo, seed)
                            cmd = [
                                *base_cmd,
                                '--edge_csv', str(edge_csv),
                                '--region', region,
                                '--time_slots', str(args.time_slots),
                                '--arrival_rate', str(point['arrival_rate']),
                                '--mean_service_time', str(args.mean_service_time),
                                '--k', str(args.k),
                                '--seed', str(seed),
                                '--algo', algo,
                                '--cap_scale', str(args.cap_scale),
                                '--out_dir', str(out_dir),
                            ]
                            if 'fixed_sfc_len' in point:
                                cmd += ['--fixed_sfc_len', str(point['fixed_sfc_len'])]
                            if 'fixed_reliability' in point:
                                cmd += ['--fixed_reliability', f"{point['fixed_reliability']:.3f}"]
                            cmd += method_cfg['extra_args']
                            tasks.append(
                                {
                                    'section': section,
                                    'experiment': experiment,
                                    'region': region,
                                    'x_name': x_name,
                                    'x_value': x_value,
                                    'method': method,
                                    'algo': algo,
                                    'seed': seed,
                                    'out_dir': out_dir,
                                    'metric_path': mpath,
                                    'cmd': cmd,
                                }
                            )
    return tasks


def run_one_task(task: Dict, skip_existing: bool, dry_run: bool) -> Dict:
    metric_file = Path(task['metric_path'])
    if skip_existing and metric_file.exists():
        metrics = load_metrics(metric_file)
        return {'task': task, 'metrics': metrics, 'status': 'cached'}

    if dry_run:
        return {'task': task, 'metrics': None, 'status': 'dry_run', 'command': ' '.join(task['cmd'])}

    started = time.time()
    completed = subprocess.run(task['cmd'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}) after {elapsed:.1f}s: {' '.join(task['cmd'])}\n\n{completed.stdout}"
        )
    metrics = load_metrics(metric_file)
    return {'task': task, 'metrics': metrics, 'status': 'ran', 'elapsed_s': elapsed}


def summarize_and_save(raw_rows: List[Dict], out_root: Path) -> None:
    raw_df = pd.DataFrame(raw_rows)
    raw_path = out_root / 'raw_metrics_long.csv'
    raw_df.to_csv(raw_path, index=False)

    fairness_cols = ['section', 'experiment', 'region', 'x_name', 'x_value', 'seed']
    fairness_df = raw_df.groupby(fairness_cols).agg(
        trace_hash_nunique=('trace_hash', 'nunique'),
        total_arrivals_nunique=('total_arrivals', 'nunique'),
        arrivals_per_slot_nunique=('arrivals_per_slot_key', 'nunique'),
    ).reset_index()
    fairness_df['trace_consistent'] = (
        (fairness_df['trace_hash_nunique'] == 1)
        & (fairness_df['total_arrivals_nunique'] == 1)
        & (fairness_df['arrivals_per_slot_nunique'] == 1)
    )
    fairness_path = out_root / 'fairness_checks.csv'
    fairness_df.to_csv(fairness_path, index=False)
    if not fairness_df['trace_consistent'].all():
        bad = fairness_df.loc[~fairness_df['trace_consistent']]
        raise RuntimeError(f'Trace consistency check failed:\n{bad.to_string(index=False)}')

    metric_cols = [
        'acceptance_ratio',
        'avg_cost_per_accepted',
        'total_cost_resource',
        'p95_e2e_delay_accepted',
        'qos_rejection_ratio',
        'avg_e2e_delay_accepted',
        'final_utilization',
        'avg_utilization_over_time',
        'avg_transmitted_bits_per_accepted',
        'avg_split_transmitted_bits_per_accepted',
        'avg_split_overhead_ratio_accepted',
        'total_arrivals',
        'total_accepted',
    ]
    agg_spec = {metric: ['mean', 'std', 'count'] for metric in metric_cols if metric in raw_df.columns}
    summary = raw_df.groupby(['section', 'experiment', 'region', 'x_name', 'x_value', 'method', 'algo']).agg(agg_spec)
    summary.columns = ['_'.join(column).rstrip('_') for column in summary.columns.to_flat_index()]
    summary = summary.reset_index()

    for metric in metric_cols:
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        count_col = f'{metric}_count'
        if mean_col in summary.columns:
            summary[f'{metric}_ci95'] = 1.96 * (summary[std_col].fillna(0.0) / summary[count_col].clip(lower=1).pow(0.5))

    summary_path = out_root / 'summary_metrics.csv'
    summary.to_csv(summary_path, index=False)

    print(f'Saved raw metrics: {raw_path}')
    print(f'Saved fairness report: {fairness_path}')
    print(f'Saved summary metrics: {summary_path}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_script', type=str, default='online_mdca_sim_delay_split_strict.py')
    parser.add_argument('--conda_env', type=str, default='SFC')
    parser.add_argument('--use_current_python', action='store_true')
    parser.add_argument('--max_workers', type=int, default=1)
    parser.add_argument('--progress_every', type=int, default=25)
    parser.add_argument('--edge_csv', type=str, default='Edge_devices.csv')
    parser.add_argument('--regions', type=str, default='Alabama,Arizona')
    parser.add_argument('--seeds', type=str, default='1,2,3,4,5')
    parser.add_argument('--time_slots', type=int, default=120)
    parser.add_argument('--mean_service_time', type=float, default=9.0)
    parser.add_argument('--cap_scale', type=float, default=0.2)
    parser.add_argument('--k', type=int, default=6)
    parser.add_argument('--load_rates', type=str, default='5,10,15,20,25,30,35,40,45,50')
    parser.add_argument('--length_arrival_rate', type=float, default=20.0)
    parser.add_argument('--reliability_arrival_rate', type=float, default=12.0)
    parser.add_argument('--sfc_lengths', type=str, default='1,2,3,4,5,6')
    parser.add_argument('--reliability_points', type=str, default='0.900,0.930,0.960,0.980,0.990,0.992,0.994,0.996,0.998,0.999')
    parser.add_argument('--section', type=str, default='all', choices=['all', 'unsplit', 'split_enabled', 'mdca', 'dmdca'])
    parser.add_argument('--experiments', type=str, default='load,length,reliability')
    parser.add_argument('--max_points_per_experiment', type=int, default=None)
    parser.add_argument('--out_root', type=str, default='./paper_online_runs_tuned')
    parser.add_argument('--pvfp_model_path', type=str, default='')
    parser.add_argument('--pvfp_disable_fallback', action='store_true')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    sim_script = Path(args.sim_script).resolve()
    edge_csv = Path(args.edge_csv).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args, sim_script, edge_csv, out_root)
    total = len(tasks)
    print(
        f"Prepared {total} runs | section={args.section} | experiments={args.experiments} | "
        f"regions={args.regions} | seeds={args.seeds} | use_current_python={args.use_current_python} | "
        f"max_workers={args.max_workers} | mean_service_time={args.mean_service_time} | cap_scale={args.cap_scale}",
        flush=True,
    )

    if args.dry_run:
        for task in tasks:
            result = run_one_task(task, args.skip_existing, dry_run=True)
            print(result['command'])
        print('Dry run complete.')
        return

    raw_rows: List[Dict] = []
    completed_count = 0
    started_all = time.time()

    def append_result(task: Dict, result: Dict) -> None:
        nonlocal completed_count
        metrics = dict(result['metrics'])
        metrics['arrivals_per_slot_key'] = json.dumps(metrics.get('arrivals_per_slot', []), separators=(',', ':'))
        row = {
            'section': task['section'],
            'experiment': task['experiment'],
            'region': task['region'],
            'x_name': task['x_name'],
            'x_value': task['x_value'],
            'method': task['method'],
            'algo': task['algo'],
            'seed': task['seed'],
        }
        row.update(metrics)
        raw_rows.append(row)
        completed_count += 1
        elapsed = time.time() - started_all
        rate = elapsed / max(completed_count, 1)
        eta = rate * (total - completed_count)
        if completed_count == total or completed_count % max(args.progress_every, 1) == 0:
            print(
                f"[{completed_count}/{total}] {task['section']}/{task['experiment']}/{task['region']}/{task['method']}/seed{task['seed']} "
                f"status={result['status']} run_s={result.get('elapsed_s', 0.0):.1f} eta_min={eta / 60.0:.1f}",
                flush=True,
            )

    if args.max_workers <= 1:
        for task in tasks:
            result = run_one_task(task, args.skip_existing, dry_run=False)
            append_result(task, result)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {
                executor.submit(run_one_task, task, args.skip_existing, False): task
                for task in tasks
            }
            for future in concurrent.futures.as_completed(future_map):
                task = future_map[future]
                result = future.result()
                append_result(task, result)

    summarize_and_save(raw_rows, out_root)


if __name__ == '__main__':
    main()
