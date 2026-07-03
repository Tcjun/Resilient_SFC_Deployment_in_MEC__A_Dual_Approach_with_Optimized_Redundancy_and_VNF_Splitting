#!/usr/bin/env python3
"""Small-scale exact reference for compact offline P1'.

The script intentionally avoids external MILP solvers. It enumerates
reliability-feasible backup configurations and solves the resulting placement
problem with branch-and-bound. Only completed searches are reported as exact
OPT; timeouts are counted separately and excluded from gap statistics.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VNF_TYPES = 6
EPS = 1e-9


@dataclass(frozen=True)
class Instance:
    req: int
    vnf: int
    demand: int
    inst_cost: float


@dataclass(frozen=True)
class RequestConfig:
    request_id: int
    backup_levels: Tuple[int, ...]
    reliability: float
    instances: Tuple[Instance, ...]
    lower_bound: float
    total_resource: int


@dataclass
class SmallInstance:
    seed: int
    j: int
    k: int
    l: int
    m: int
    chains: List[List[int]]
    demand: np.ndarray
    inst_cost: np.ndarray
    reliability: np.ndarray
    req_reliability: np.ndarray
    unit_cost: np.ndarray
    capacities: np.ndarray


class ExactTimeout(RuntimeError):
    pass


def parse_csv_ints(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_scales(value: str) -> List[Tuple[int, int, int]]:
    scales: List[Tuple[int, int, int]] = []
    for part in value.split(","):
        token = part.strip().lower()
        if not token:
            continue
        fields = token.split("x")
        if len(fields) != 3:
            raise ValueError(f"Invalid scale '{part}', expected JxKxL")
        scales.append((int(fields[0]), int(fields[1]), int(fields[2])))
    if not scales:
        raise ValueError("At least one scale is required")
    return scales


def stage_reliability(r: float, backup_count: int) -> float:
    r = float(np.clip(r, 0.0, 0.999999999999))
    b = max(0, int(backup_count))
    return float(1.0 - (1.0 - r) ** (b + 1))


def request_reliability(chain: Sequence[int], rel_row: np.ndarray, backup_levels: Sequence[int]) -> float:
    result = 1.0
    for pos, vnf in enumerate(chain):
        result *= stage_reliability(float(rel_row[int(vnf)]), int(backup_levels[pos]))
    return float(result)


def config_instances(
    request_id: int,
    chain: Sequence[int],
    demand: np.ndarray,
    inst_cost: np.ndarray,
    backup_levels: Sequence[int],
) -> Tuple[Instance, ...]:
    instances: List[Instance] = []
    for pos, vnf in enumerate(chain):
        v = int(vnf)
        copies = int(backup_levels[pos]) + 1
        for _ in range(copies):
            instances.append(
                Instance(
                    req=int(request_id),
                    vnf=v,
                    demand=int(demand[request_id, v]),
                    inst_cost=float(inst_cost[request_id, v]),
                )
            )
    instances.sort(key=lambda item: (item.demand, item.inst_cost), reverse=True)
    return tuple(instances)


def enumerate_request_configs(instance: SmallInstance, request_id: int) -> List[RequestConfig]:
    chain = instance.chains[request_id]
    min_delta = float(np.min(instance.unit_cost))
    configs: List[RequestConfig] = []
    for levels in product(range(instance.m + 1), repeat=len(chain)):
        rel = request_reliability(chain, instance.reliability[request_id], levels)
        if rel + EPS < float(instance.req_reliability[request_id]):
            continue
        instances = config_instances(request_id, chain, instance.demand, instance.inst_cost, levels)
        lb = sum(item.inst_cost + item.demand * min_delta for item in instances)
        total_res = sum(item.demand for item in instances)
        configs.append(
            RequestConfig(
                request_id=int(request_id),
                backup_levels=tuple(int(x) for x in levels),
                reliability=float(rel),
                instances=instances,
                lower_bound=float(lb),
                total_resource=int(total_res),
            )
        )
    configs.sort(key=lambda cfg: (cfg.lower_bound, cfg.total_resource, cfg.backup_levels))
    return configs


def select_mdca_config(instance: SmallInstance, request_id: int) -> Optional[RequestConfig]:
    chain = instance.chains[request_id]
    levels = [0 for _ in chain]
    current = request_reliability(chain, instance.reliability[request_id], levels)
    while current + EPS < float(instance.req_reliability[request_id]):
        best: Optional[Tuple[float, int, float]] = None
        for pos, vnf in enumerate(chain):
            if levels[pos] >= instance.m:
                continue
            old_stage = stage_reliability(float(instance.reliability[request_id, int(vnf)]), levels[pos])
            new_stage = stage_reliability(float(instance.reliability[request_id, int(vnf)]), levels[pos] + 1)
            if new_stage <= old_stage + EPS:
                continue
            new_rel = current / max(old_stage, EPS) * new_stage
            gain = new_rel - current
            ratio = gain / max(float(instance.demand[request_id, int(vnf)]), EPS)
            if best is None or ratio > best[0]:
                best = (float(ratio), int(pos), float(new_rel))
        if best is None:
            return None
        _ratio, pos, new_rel = best
        levels[pos] += 1
        current = new_rel
    instances = config_instances(request_id, chain, instance.demand, instance.inst_cost, levels)
    min_delta = float(np.min(instance.unit_cost))
    lb = sum(item.inst_cost + item.demand * min_delta for item in instances)
    total_res = sum(item.demand for item in instances)
    return RequestConfig(
        request_id=int(request_id),
        backup_levels=tuple(levels),
        reliability=float(current),
        instances=instances,
        lower_bound=float(lb),
        total_resource=int(total_res),
    )


def greedy_mdca_cost(instance: SmallInstance, configs: Sequence[RequestConfig]) -> Tuple[bool, float]:
    cap = instance.capacities.astype(int).copy()
    cloud_order = list(np.argsort(instance.unit_cost, kind="stable").astype(int))
    cost = 0.0
    req_order = sorted(configs, key=lambda cfg: (cfg.total_resource, cfg.request_id))
    for cfg in req_order:
        snapshot = cap.copy()
        local_cost = 0.0
        feasible = True
        for item in sorted(cfg.instances, key=lambda inst: (inst.demand, inst.inst_cost), reverse=True):
            placed = False
            for cloudlet in cloud_order:
                if cap[cloudlet] >= item.demand:
                    cap[cloudlet] -= item.demand
                    local_cost += item.inst_cost + item.demand * float(instance.unit_cost[cloudlet])
                    placed = True
                    break
            if not placed:
                feasible = False
                break
        if not feasible:
            cap = snapshot
            return False, math.nan
        cost += local_cost
    return True, float(cost)


def build_small_instance(seed: int, j: int, k: int, l: int, m: int, capacity_factor: float) -> SmallInstance:
    rng = np.random.default_rng(seed * 1000003 + j * 10007 + k * 101 + l)
    chains: List[List[int]] = []
    demand = np.zeros((j, VNF_TYPES), dtype=int)
    inst_cost = np.zeros((j, VNF_TYPES), dtype=float)
    reliability = np.zeros((j, VNF_TYPES), dtype=float)
    req_reliability = 0.90 + 0.09 * rng.random(j)
    for req in range(j):
        chain = list(rng.choice(VNF_TYPES, size=l, replace=False).astype(int))
        rng.shuffle(chain)
        chains.append(chain)
        for vnf in chain:
            demand[req, vnf] = int(rng.integers(40, 401))
            inst_cost[req, vnf] = float(rng.integers(1, 4))
            reliability[req, vnf] = float(0.90 + 0.09 * rng.random())

    unit_cost = 0.01 + 0.04 * rng.random(k)
    provisional = SmallInstance(
        seed=seed,
        j=j,
        k=k,
        l=l,
        m=m,
        chains=chains,
        demand=demand,
        inst_cost=inst_cost,
        reliability=reliability,
        req_reliability=req_reliability,
        unit_cost=unit_cost,
        capacities=np.ones(k, dtype=int),
    )
    mdca_configs = [select_mdca_config(provisional, req) for req in range(j)]
    selected_resource = sum(cfg.total_resource for cfg in mdca_configs if cfg is not None)
    if selected_resource <= 0:
        selected_resource = int(np.sum(demand))
    target_total = max(float(selected_resource) * float(capacity_factor), float(np.max(demand)) * k)
    weights = 0.8 + 0.4 * rng.random(k)
    capacities = np.ceil(target_total * weights / np.sum(weights)).astype(int)
    max_demand = int(np.max(demand)) if int(np.max(demand)) > 0 else 1
    capacities = np.maximum(capacities, max_demand)
    return SmallInstance(
        seed=seed,
        j=j,
        k=k,
        l=l,
        m=m,
        chains=chains,
        demand=demand,
        inst_cost=inst_cost,
        reliability=reliability,
        req_reliability=req_reliability,
        unit_cost=unit_cost,
        capacities=capacities.astype(int),
    )


def exact_opt_cost(
    instance: SmallInstance,
    request_configs: Sequence[Sequence[RequestConfig]],
    mdca_upper_bound: float,
    timeout_seconds: float,
) -> Dict[str, float | int | str]:
    start = time.perf_counter()
    deadline = start + float(timeout_seconds)
    ordered = sorted(range(instance.j), key=lambda req: (len(request_configs[req]), req))
    configs_ordered = [list(request_configs[req]) for req in ordered]
    min_lb = [min(cfg.lower_bound for cfg in cfgs) for cfgs in configs_ordered]
    suffix_lb = [0.0] * (len(min_lb) + 1)
    for idx in range(len(min_lb) - 1, -1, -1):
        suffix_lb[idx] = suffix_lb[idx + 1] + min_lb[idx]

    best = float(mdca_upper_bound) if math.isfinite(mdca_upper_bound) else math.inf
    nodes = 0
    seen_req: Dict[Tuple[int, Tuple[int, ...]], float] = {}
    cloud_order = list(np.argsort(instance.unit_cost, kind="stable").astype(int))

    def check_timeout() -> None:
        if time.perf_counter() > deadline:
            raise ExactTimeout()

    def recurse_request(pos: int, cap_tuple: Tuple[int, ...], cost_so_far: float) -> None:
        nonlocal best, nodes
        nodes += 1
        if (nodes & 4095) == 0:
            check_timeout()
        if cost_so_far + suffix_lb[pos] >= best - EPS:
            return
        key = (pos, cap_tuple)
        previous = seen_req.get(key)
        if previous is not None and previous <= cost_so_far + EPS:
            return
        seen_req[key] = cost_so_far
        if pos == len(configs_ordered):
            best = min(best, cost_so_far)
            return

        for cfg in configs_ordered[pos]:
            if cost_so_far + cfg.lower_bound + suffix_lb[pos + 1] >= best - EPS:
                continue
            instances = cfg.instances
            inst_suffix = [0.0] * (len(instances) + 1)
            min_delta = float(np.min(instance.unit_cost))
            for inst_idx in range(len(instances) - 1, -1, -1):
                item = instances[inst_idx]
                inst_suffix[inst_idx] = inst_suffix[inst_idx + 1] + item.inst_cost + item.demand * min_delta

            placement_seen: Dict[Tuple[int, Tuple[int, ...]], float] = {}
            cap = list(cap_tuple)

            def place_instance(inst_idx: int, add_cost: float) -> None:
                nonlocal nodes
                nodes += 1
                if (nodes & 4095) == 0:
                    check_timeout()
                if cost_so_far + add_cost + inst_suffix[inst_idx] + suffix_lb[pos + 1] >= best - EPS:
                    return
                state = (inst_idx, tuple(cap))
                previous_add = placement_seen.get(state)
                if previous_add is not None and previous_add <= add_cost + EPS:
                    return
                placement_seen[state] = add_cost
                if inst_idx == len(instances):
                    recurse_request(pos + 1, tuple(cap), cost_so_far + add_cost)
                    return

                item = instances[inst_idx]
                for cloudlet in cloud_order:
                    if cap[cloudlet] < item.demand:
                        continue
                    cap[cloudlet] -= item.demand
                    place_instance(
                        inst_idx + 1,
                        add_cost + item.inst_cost + item.demand * float(instance.unit_cost[cloudlet]),
                    )
                    cap[cloudlet] += item.demand

            place_instance(0, 0.0)

    try:
        recurse_request(0, tuple(int(x) for x in instance.capacities), 0.0)
        runtime = time.perf_counter() - start
        if not math.isfinite(best):
            return {"status": "infeasible", "opt_cost": math.nan, "runtime_s": runtime, "nodes": nodes}
        return {"status": "solved", "opt_cost": float(best), "runtime_s": runtime, "nodes": nodes}
    except ExactTimeout:
        runtime = time.perf_counter() - start
        incumbent = float(best) if math.isfinite(best) else math.nan
        return {"status": "timeout", "opt_cost": math.nan, "incumbent_cost": incumbent, "runtime_s": runtime, "nodes": nodes}


def solve_one(seed: int, j: int, k: int, l: int, m: int, capacity_factor: float, timeout_seconds: float) -> Dict:
    instance = build_small_instance(seed=seed, j=j, k=k, l=l, m=m, capacity_factor=capacity_factor)
    scale = f"{j}x{k}x{l}"
    request_configs = [enumerate_request_configs(instance, req) for req in range(j)]
    lower_bound = sum((min((cfg.lower_bound for cfg in cfgs), default=math.nan) for cfgs in request_configs))

    if any(len(cfgs) == 0 for cfgs in request_configs):
        return {
            "scale": scale,
            "J": j,
            "K": k,
            "L": l,
            "M": m,
            "seed": seed,
            "status": "infeasible",
            "lower_bound": lower_bound,
            "opt_cost": math.nan,
            "mdca_cost": math.nan,
            "mdca_opt_ratio": math.nan,
            "runtime_s": 0.0,
            "nodes": 0,
            "mdca_feasible": False,
            "sanity_lb_le_opt": False,
            "sanity_opt_le_mdca": False,
        }

    mdca_configs = [select_mdca_config(instance, req) for req in range(j)]
    mdca_feasible = all(cfg is not None for cfg in mdca_configs)
    mdca_cost = math.nan
    if mdca_feasible:
        mdca_feasible, mdca_cost = greedy_mdca_cost(instance, [cfg for cfg in mdca_configs if cfg is not None])

    exact = exact_opt_cost(
        instance=instance,
        request_configs=request_configs,
        mdca_upper_bound=mdca_cost if mdca_feasible else math.inf,
        timeout_seconds=timeout_seconds,
    )
    status = str(exact["status"])
    opt_cost = float(exact.get("opt_cost", math.nan))
    ratio = (float(mdca_cost) / opt_cost) if status == "solved" and mdca_feasible and opt_cost > 0 else math.nan
    lb_ok = bool(status == "solved" and lower_bound <= opt_cost + 1e-6)
    opt_ok = bool(status == "solved" and mdca_feasible and opt_cost <= float(mdca_cost) + 1e-6)
    if status == "solved" and (not lb_ok or (mdca_feasible and not opt_ok)):
        raise AssertionError(
            f"Sanity check failed for scale={scale} seed={seed}: "
            f"LB={lower_bound}, OPT={opt_cost}, MDCA={mdca_cost}"
        )

    return {
        "scale": scale,
        "J": j,
        "K": k,
        "L": l,
        "M": m,
        "seed": seed,
        "status": status,
        "lower_bound": float(lower_bound),
        "opt_cost": opt_cost,
        "mdca_cost": float(mdca_cost) if mdca_feasible else math.nan,
        "mdca_opt_ratio": ratio,
        "runtime_s": float(exact.get("runtime_s", math.nan)),
        "nodes": int(exact.get("nodes", 0)),
        "mdca_feasible": bool(mdca_feasible),
        "sanity_lb_le_opt": lb_ok,
        "sanity_opt_le_mdca": opt_ok,
        "incumbent_cost": float(exact.get("incumbent_cost", math.nan)),
        "capacity_total": int(np.sum(instance.capacities)),
        "min_unit_cost": float(np.min(instance.unit_cost)),
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for keys, group in raw.groupby(["scale", "J", "K", "L", "M"], sort=False):
        scale, j, k, l, m = keys
        solved = group[group["status"] == "solved"].copy()
        ratio = solved["mdca_opt_ratio"].dropna()
        rows.append(
            {
                "scale": scale,
                "J": int(j),
                "K": int(k),
                "L": int(l),
                "M": int(m),
                "total_runs": int(len(group)),
                "solved_instances": int(len(solved)),
                "ratio_instances": int(len(ratio)),
                "timeout_count": int((group["status"] == "timeout").sum()),
                "infeasible_count": int((group["status"] == "infeasible").sum()),
                "mdca_infeasible_count": int((~group["mdca_feasible"].astype(bool)).sum()),
                "mean_mdca_opt_ratio": float(ratio.mean()) if not ratio.empty else math.nan,
                "max_mdca_opt_ratio": float(ratio.max()) if not ratio.empty else math.nan,
                "mean_opt_runtime_s": float(solved["runtime_s"].mean()) if not solved.empty else math.nan,
                "mean_lb_opt_ratio": float((solved["lower_bound"] / solved["opt_cost"]).mean()) if not solved.empty else math.nan,
            }
        )
    return pd.DataFrame(rows)


def write_latex_rows(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "% Generated by run_small_scale_opt_reference.py",
        "% scale & solved/total & ratio instances & timeout+infeasible & mean MDCA/OPT & max MDCA/OPT & mean runtime (s) \\\\",
    ]
    for row in summary.itertuples(index=False):
        unsolved = int(row.timeout_count) + int(row.infeasible_count)
        mean_ratio = "--" if math.isnan(float(row.mean_mdca_opt_ratio)) else f"{float(row.mean_mdca_opt_ratio):.4f}"
        max_ratio = "--" if math.isnan(float(row.max_mdca_opt_ratio)) else f"{float(row.max_mdca_opt_ratio):.4f}"
        runtime = "--" if math.isnan(float(row.mean_opt_runtime_s)) else f"{float(row.mean_opt_runtime_s):.3f}"
        lines.append(
            f"{row.scale} & {int(row.solved_instances)}/{int(row.total_runs)} & "
            f"{int(row.ratio_instances)} & {unsolved} & {mean_ratio} & {max_ratio} & {runtime} \\\\"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--scales", type=str, default="3x5x2,5x5x2,5x8x3", help="Comma-separated JxKxL scales")
    parser.add_argument("--out_dir", type=str, default="paper_data/small_scale_opt_reference")
    parser.add_argument("--max_backup_level", type=int, default=2)
    parser.add_argument("--timeout_seconds", type=float, default=30.0)
    parser.add_argument("--capacity_factor", type=float, default=1.35)
    parser.add_argument("--max_workers", type=int, default=1)
    args = parser.parse_args()

    seeds = parse_csv_ints(args.seeds)
    scales = parse_scales(args.scales)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for scale_idx, (j, k, l) in enumerate(scales):
        if l > VNF_TYPES:
            raise ValueError(f"L={l} exceeds VNF_TYPES={VNF_TYPES}")
        for seed in seeds:
            tasks.append(
                {
                    "scale_idx": int(scale_idx),
                    "seed": int(seed),
                    "j": int(j),
                    "k": int(k),
                    "l": int(l),
                }
            )

    rows_with_order = []

    def record_result(order: Tuple[int, int], row: Dict) -> None:
        rows_with_order.append((order, row))
        print(
            f"{row['scale']} seed={row['seed']} status={row['status']} "
            f"ratio={row['mdca_opt_ratio'] if not math.isnan(row['mdca_opt_ratio']) else 'NA'} "
            f"runtime={row['runtime_s']:.3f}s",
            flush=True,
        )

    max_workers = max(1, int(args.max_workers))
    if max_workers == 1:
        for task in tasks:
            row = solve_one(
                seed=task["seed"],
                j=task["j"],
                k=task["k"],
                l=task["l"],
                m=int(args.max_backup_level),
                capacity_factor=float(args.capacity_factor),
                timeout_seconds=float(args.timeout_seconds),
            )
            record_result((task["scale_idx"], task["seed"]), row)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    solve_one,
                    seed=task["seed"],
                    j=task["j"],
                    k=task["k"],
                    l=task["l"],
                    m=int(args.max_backup_level),
                    capacity_factor=float(args.capacity_factor),
                    timeout_seconds=float(args.timeout_seconds),
                ): task
                for task in tasks
            }
            for future in concurrent.futures.as_completed(future_map):
                task = future_map[future]
                row = future.result()
                record_result((task["scale_idx"], task["seed"]), row)

    rows = [row for _order, row in sorted(rows_with_order, key=lambda item: item[0])]
    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    raw_path = out_dir / "raw.csv"
    summary_path = out_dir / "summary.csv"
    latex_path = out_dir / "small_scale_opt_reference_table.tex"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_latex_rows(summary, latex_path)
    print(f"Saved raw results: {raw_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved LaTeX rows: {latex_path}")


if __name__ == "__main__":
    main()
