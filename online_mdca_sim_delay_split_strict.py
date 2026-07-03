#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pvfp_fdrl_adapter import PvfpFdrlConfig, PvfpFdrlPlacementPolicy



@dataclass(frozen=True)
class AlgorithmSpec:
    algo: str
    request_order: str
    backup_policy: str
    cloudlet_policy: str
    instance_order: str
    distinct_request_cloudlets: bool
    default_max_backup_level: int


def get_algorithm_spec(algo: str) -> AlgorithmSpec:
    algo_key = algo.lower()
    if algo_key == "mdca":
        return AlgorithmSpec(
            algo="mdca",
            request_order="resource_asc",
            backup_policy="gain_ratio",
            cloudlet_policy="cheapest",
            instance_order="resource_desc",
            distinct_request_cloudlets=False,
            default_max_backup_level=2,
        )
    if algo_key == "dtsp":
        return AlgorithmSpec(
            algo="dtsp",
            request_order="reliability_asc",
            backup_policy="resource_reliability_round_robin",
            cloudlet_policy="cheapest",
            instance_order="original",
            distinct_request_cloudlets=True,
            default_max_backup_level=2,
        )
    if algo_key == "hspa":
        return AlgorithmSpec(
            algo="hspa",
            request_order="pay_desc",
            backup_policy="low_reliability_first",
            cloudlet_policy="cheapest",
            instance_order="original",
            distinct_request_cloudlets=False,
            default_max_backup_level=2,
        )
    if algo_key == "random":
        return AlgorithmSpec(
            algo="random",
            request_order="random",
            backup_policy="random_sequential",
            cloudlet_policy="random",
            instance_order="original",
            distinct_request_cloudlets=False,
            default_max_backup_level=2,
        )
    if algo_key == "pvfp_fdrl":
        return AlgorithmSpec(
            algo="pvfp_fdrl",
            request_order="resource_asc",
            backup_policy="gain_ratio",
            cloudlet_policy="pvfp_fdrl",
            instance_order="resource_desc",
            distinct_request_cloudlets=False,
            default_max_backup_level=2,
        )
    raise ValueError(f"Unsupported algo={algo}. Expected one of mdca/dtsp/hspa/random/pvfp_fdrl.")


def derive_seed(namespace: str, *parts: Any) -> int:
    payload = "|".join([namespace, *[str(p) for p in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def update_trace_hash(hasher: "hashlib._Hash", slot: int, arrays: Dict[str, Any]) -> None:
    hasher.update(np.asarray([slot], dtype=np.int64).tobytes())
    for key in sorted(arrays):
        value = arrays[key]
        hasher.update(key.encode("utf-8"))
        if isinstance(value, list):
            value = np.asarray(value, dtype=np.int64)
        elif isinstance(value, tuple):
            value = np.asarray(list(value), dtype=np.int64)
        elif not isinstance(value, np.ndarray):
            value = np.asarray([value])
        hasher.update(np.asarray(value).tobytes())


def generate_sfc_batch(
    rng: np.random.Generator,
    n_arrivals: int,
    k: int,
    fixed_sfc_len: Optional[int] = None,
) -> Tuple[np.ndarray, List[List[int]]]:
    sfc = np.zeros((n_arrivals, k), dtype=int)
    chains: List[List[int]] = []

    for i_req in range(n_arrivals):
        if fixed_sfc_len is None:
            row = rng.integers(0, 2, size=k, dtype=int)
            if int(row.sum()) == 0:
                row[int(rng.integers(0, k))] = 1
            v_list = list(np.where(row == 1)[0].astype(int))
            rng.shuffle(v_list)
            sfc[i_req, :] = row
            chains.append(v_list)
            continue

        sfc_len = int(max(1, min(k, fixed_sfc_len)))
        v_list = list(rng.choice(k, size=sfc_len, replace=False).astype(int))
        rng.shuffle(v_list)
        sfc[i_req, v_list] = 1
        chains.append(v_list)

    return sfc, chains

# -----------------------------
# Delay model helpers (placement-aware)
# -----------------------------
def build_latency_and_bandwidth(
    rng: np.random.Generator,
    m: int,
    df_cloudlets: Optional[pd.DataFrame] = None,
    fiber_speed_mps: float = 2e8,
    switch_delay_s: float = 2e-4,
    bw_min_bps: float = 1e9,
    bw_max_bps: float = 1e10,
    # --- stricter/backhaul-realistic options ---
    knn_k: int = 4,
    use_device_bandwidth: bool = True,
    device_bw_scale: float = 1.0,
):
    """
    Build a placement-aware inter-cloudlet propagation-latency matrix L (seconds)
    and an effective backhaul-bandwidth matrix BW (bits/s).

    Strict version (recommended for revisions):
      - L is derived from geographical distance (lat/lon) -> fiber propagation + per-hop switch delay.
      - BW is derived from Edge_devices.csv per-cloudlet "Bandwidth (MB/s)" (bottleneck of endpoints),
        instead of being fully random.
      - The backhaul is NOT assumed to be a complete graph: we construct a k-NN topology in geographic
        space and compute all-pairs *shortest-propagation-delay* paths. For each pair (u,v),
        L[u,v] is the total propagation delay along the chosen path, and BW[u,v] is the bottleneck
        bandwidth along that path.

    If coordinates or bandwidth are unavailable, the function falls back to the original behavior
    (symmetric random L and BW) to keep the simulator runnable.

    Notes:
      - This matrix models wired/fiber backhaul between cloudlets, NOT the UE access link.
      - The diagonal is set to 0 latency and infinite bandwidth (same cloudlet).
    """
    # -----------------------------
    # Helpers
    # -----------------------------
    def _infer_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        cols_lower = {c.lower(): c for c in df.columns}
        for key in candidates:
            if key.lower() in cols_lower:
                return cols_lower[key.lower()]
        return None

    def _compute_dist_m(df: pd.DataFrame) -> Optional[np.ndarray]:
        lat_col = _infer_col(df, ["Latitude", "lat"])
        lon_col = _infer_col(df, ["Longitude", "lon", "lng"])
        if lat_col is None or lon_col is None:
            return None
        lats = np.deg2rad(df[lat_col].to_numpy(dtype=float))
        lons = np.deg2rad(df[lon_col].to_numpy(dtype=float))
        dlat = lats[:, None] - lats[None, :]
        dlon = lons[:, None] - lons[None, :]
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        R = 6_371_000.0
        return R * c  # meters

    def _device_bw_bps(df: pd.DataFrame) -> Optional[np.ndarray]:
        if not use_device_bandwidth:
            return None
        bw_col = _infer_col(df, ["Bandwidth (MB/s)", "bandwidth (mb/s)", "bandwidth"])
        if bw_col is None:
            return None
        bw_mb_s = df[bw_col].to_numpy(dtype=float)
        # MB/s -> bits/s (assume 1MB = 1e6 bytes)
        bw_bps = bw_mb_s * 8e6
        bw_bps = np.maximum(bw_bps, 1.0) * float(device_bw_scale)
        return bw_bps

    # -----------------------------
    # 1) Distance-based backhaul topology
    # -----------------------------
    if df_cloudlets is None:
        # fallback: symmetric random L and BW (original behavior)
        L = rng.uniform(0.001, 0.010, size=(m, m))
        L = (L + L.T) / 2.0
        np.fill_diagonal(L, 0.0)
        BW = rng.uniform(float(bw_min_bps), float(bw_max_bps), size=(m, m))
        BW = (BW + BW.T) / 2.0
        np.fill_diagonal(BW, np.inf)
        return L, BW

    dist_m = _compute_dist_m(df_cloudlets)
    bw_node = _device_bw_bps(df_cloudlets)

    if dist_m is None:
        # no coordinates -> fallback to symmetric random L
        L = rng.uniform(0.001, 0.010, size=(m, m))
        L = (L + L.T) / 2.0
        np.fill_diagonal(L, 0.0)
        if bw_node is None:
            BW = rng.uniform(float(bw_min_bps), float(bw_max_bps), size=(m, m))
            BW = (BW + BW.T) / 2.0
        else:
            BW = np.minimum(bw_node[:, None], bw_node[None, :])
        np.fill_diagonal(BW, np.inf)
        return L, BW

    # Edge latency (propagation + per-hop switch)
    edge_lat = dist_m / float(fiber_speed_mps) + float(switch_delay_s)
    np.fill_diagonal(edge_lat, 0.0)

    # Edge bandwidth: either from devices or random
    if bw_node is None:
        edge_bw = rng.uniform(float(bw_min_bps), float(bw_max_bps), size=(m, m))
        edge_bw = (edge_bw + edge_bw.T) / 2.0
    else:
        edge_bw = np.minimum(bw_node[:, None], bw_node[None, :])

    np.fill_diagonal(edge_bw, np.inf)

    # -----------------------------
    # 2) Build k-NN adjacency (undirected)
    # -----------------------------
    k = int(knn_k)
    if k <= 0 or k >= m:
        # complete graph
        L = edge_lat.copy()
        BW = edge_bw.copy()
        np.fill_diagonal(L, 0.0)
        np.fill_diagonal(BW, np.inf)
        return L, BW

    adj = np.zeros((m, m), dtype=bool)
    for i in range(m):
        # nearest neighbors by geographic distance (exclude itself)
        nn = np.argsort(dist_m[i, :])
        nn = [j for j in nn if j != i][:k]
        for j in nn:
            adj[i, j] = True
            adj[j, i] = True

    # Ensure connectivity by connecting components with closest inter-component edge if needed
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(m):
        for j in range(i + 1, m):
            if adj[i, j]:
                union(i, j)

    # connect components until single component
    def components():
        comp = {}
        for i in range(m):
            r = find(i)
            comp.setdefault(r, []).append(i)
        return list(comp.values())

    comps = components()
    while len(comps) > 1:
        # find closest pair across any two components
        best = (None, None, float("inf"))
        for a in comps[0]:
            # compare to all nodes outside comp0
            for ci in range(1, len(comps)):
                for b in comps[ci]:
                    d = dist_m[a, b]
                    if d < best[2]:
                        best = (a, b, d)
        a, b, _d = best
        adj[a, b] = True
        adj[b, a] = True
        union(a, b)
        comps = components()

    # Build sparse edge matrices
    sparse_lat = np.full((m, m), np.inf, dtype=float)
    sparse_bw = np.full((m, m), 1.0, dtype=float)  # avoid zero-division
    for i in range(m):
        sparse_lat[i, i] = 0.0
        sparse_bw[i, i] = np.inf

    for i in range(m):
        js = np.where(adj[i])[0]
        for j in js:
            sparse_lat[i, j] = edge_lat[i, j]
            sparse_bw[i, j] = edge_bw[i, j]

    # -----------------------------
    # 3) All-pairs shortest propagation-delay paths + bottleneck BW along chosen path
    # -----------------------------
    import heapq

    L = np.full((m, m), np.inf, dtype=float)
    BW = np.full((m, m), 1.0, dtype=float)

    for src in range(m):
        dist = np.full(m, np.inf, dtype=float)
        prev = np.full(m, -1, dtype=int)
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d_u, u = heapq.heappop(pq)
            if d_u != dist[u]:
                continue
            for v in np.where(adj[u])[0]:
                nd = d_u + sparse_lat[u, v]
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        # fill matrices for this src
        for dst in range(m):
            if dst == src:
                L[src, dst] = 0.0
                BW[src, dst] = np.inf
                continue
            if not np.isfinite(dist[dst]):
                continue
            L[src, dst] = float(dist[dst])

            # compute bottleneck bw on the chosen shortest-latency path
            bneck = float("inf")
            cur = dst
            while prev[cur] != -1:
                p = int(prev[cur])
                bneck = min(bneck, float(sparse_bw[p, cur]))
                cur = p
            if not np.isfinite(bneck) or bneck <= 0:
                bneck = 1.0
            BW[src, dst] = bneck

    # Symmetrize numerically (graph is undirected, but Dijkstra rounding may differ slightly)
    L = (L + L.T) / 2.0
    BW = (BW + BW.T) / 2.0
    np.fill_diagonal(L, 0.0)
    np.fill_diagonal(BW, np.inf)
    return L, BW


def split_workload_shares(replica_count: int, share_spec: Optional[str] = None) -> np.ndarray:
    replica_count = max(1, int(replica_count))
    if share_spec is None or str(share_spec).strip() == "" or str(share_spec).strip().lower() == "equal":
        return np.full(replica_count, 1.0 / float(replica_count), dtype=float)
    parts = [float(x.strip()) for x in str(share_spec).split(",") if x.strip()]
    if len(parts) != replica_count:
        raise ValueError("--split_workload_shares must provide one positive value per split replica")
    shares = np.asarray(parts, dtype=float)
    if np.any(shares <= 0.0):
        raise ValueError("--split_workload_shares values must be positive")
    total = float(np.sum(shares))
    if total <= 0.0:
        raise ValueError("--split_workload_shares must have a positive sum")
    return shares / total


def split_total_overhead_ratio(replica_count: int) -> float:
    replica_count = max(1, int(replica_count))
    if replica_count <= 1:
        return 0.0
    return float(replica_count - 1) / float(replica_count + 1)


def split_coordination_cycles(base_cycles: float, replica_count: int) -> float:
    return float(base_cycles) * split_total_overhead_ratio(replica_count)


def split_replica_resource_needs(
    base_need: float,
    replica_count: int,
    strict_full_replica: bool = False,
    strict_scale: float = 1.0,
    share_spec: Optional[str] = None,
) -> np.ndarray:
    base_need = float(base_need)
    replica_count = max(1, int(replica_count))
    if replica_count <= 1:
        return np.asarray([base_need], dtype=float)
    if strict_full_replica:
        return np.full(replica_count, base_need * float(strict_scale), dtype=float)

    shares = split_workload_shares(replica_count, share_spec)
    sync_reserve = base_need * split_total_overhead_ratio(replica_count)
    return shares * base_need + (sync_reserve / float(replica_count))


def split_stage_total_resource(
    base_need: float,
    replica_count: int,
    strict_full_replica: bool = False,
    strict_scale: float = 1.0,
    share_spec: Optional[str] = None,
) -> float:
    return float(np.sum(split_replica_resource_needs(base_need, replica_count, strict_full_replica, strict_scale, share_spec)))


def split_replica_intrinsic_reliabilities(
    base_r: float,
    replica_count: int,
    mode: str = "identical_active_pool",
) -> np.ndarray:
    replica_count = max(1, int(replica_count))
    base_r = float(np.clip(base_r, 1e-4, 0.9999))
    if replica_count <= 1:
        return np.asarray([base_r], dtype=float)
    if mode == "identical_active_pool":
        return np.full(replica_count, base_r, dtype=float)
    if mode != "heterogeneous_upper_bound":
        raise ValueError(f"Unsupported split reliability mode: {mode}")
    if replica_count == 2:
        delta = min(0.05, base_r - 1e-4, 0.9999 - base_r)
        return np.asarray(
            [
                float(np.clip(base_r + delta, 1e-4, 0.9999)),
                float(np.clip(base_r - delta, 1e-4, 0.9999)),
            ],
            dtype=float,
        )

    offsets = np.linspace(-0.05, 0.05, replica_count, dtype=float)
    rel = np.clip(base_r + offsets, 1e-4, 0.9999)
    return rel.astype(float)


def split_active_reliability(
    base_r: float,
    replica_count: int,
    mode: str = "identical_active_pool",
    recovery_coverage: float = 1.0,
) -> float:
    rel = split_replica_intrinsic_reliabilities(base_r, replica_count, mode=mode)
    all_survive = float(np.prod(rel))
    at_least_one_survives = float(1.0 - np.prod(1.0 - rel))
    eta = float(np.clip(recovery_coverage, 0.0, 1.0))
    return float(all_survive + eta * (at_least_one_survives - all_survive))


def stage_reliability_with_backups(active_reliability: float, backup_r: float, backup_count: int) -> float:
    active_reliability = float(np.clip(active_reliability, 0.0, 0.999999999999))
    backup_r = float(np.clip(backup_r, 0.0, 0.999999999999))
    backup_count = max(0, int(backup_count))
    return float(1.0 - (1.0 - active_reliability) * ((1.0 - backup_r) ** backup_count))


def request_reliability_with_backups(
    primary_vnfs: np.ndarray,
    active_reliability_row: np.ndarray,
    base_reliability_row: np.ndarray,
    backup_levels: Dict[int, int],
) -> float:
    req_rel = 1.0
    for v in primary_vnfs:
        v_id = int(v)
        stage_rel = stage_reliability_with_backups(
            active_reliability=float(active_reliability_row[v_id]),
            backup_r=float(base_reliability_row[v_id]),
            backup_count=int(backup_levels.get(v_id, 0)),
        )
        req_rel *= stage_rel
    return float(req_rel)


def build_split_stage_profile(
    base_need: float,
    base_cycles: float,
    replica_count: int,
    split_delay_overhead_ratio: float,
    split_coord_delay_s: float,
    split_merge_delay_s: float,
    split_consistency_delay_s: float,
    strict_full_replica: bool = False,
    strict_scale: float = 1.0,
    share_spec: Optional[str] = None,
) -> Dict[str, Any]:
    replica_count = max(1, int(replica_count))
    base_need = float(base_need)
    base_cycles = float(base_cycles)
    if base_need <= 0.0:
        inf_arr = np.full(replica_count, float("inf"), dtype=float)
        return {
            "shares": split_workload_shares(replica_count, share_spec),
            "replica_needs": inf_arr,
            "workload_factors": np.ones(replica_count, dtype=float),
            "proc_terms": inf_arr,
            "coord_cycles": float("inf"),
            "base_proc": float("inf"),
            "max_proc": float("inf"),
            "coord_compute_time": float("inf"),
            "stage_total": float("inf"),
        }

    base_proc = float(base_cycles / (base_need * 1e6))
    shares = split_workload_shares(replica_count, share_spec)
    replica_needs = split_replica_resource_needs(
        base_need=base_need,
        replica_count=replica_count,
        strict_full_replica=strict_full_replica,
        strict_scale=strict_scale,
        share_spec=share_spec,
    )
    workload_factors = np.ones(replica_count, dtype=float) if strict_full_replica else shares.copy()
    proc_terms = (workload_factors * base_cycles) / (np.maximum(replica_needs, 1e-12) * 1e6)
    max_proc = float(np.max(proc_terms))

    if replica_count <= 1:
        coord_cycles = 0.0
        coord_compute_time = 0.0
    elif strict_full_replica:
        coord_cycles = float("nan")
        coord_compute_time = float(
            split_coord_delay_s * (replica_count - 1)
            + split_merge_delay_s
            + split_consistency_delay_s
            + base_proc * float(split_delay_overhead_ratio) * (replica_count - 1)
        )
    else:
        coord_cycles = split_coordination_cycles(base_cycles, replica_count)
        coord_compute_time = float(coord_cycles / (np.sum(replica_needs) * 1e6))

    return {
        "shares": shares,
        "replica_needs": replica_needs,
        "workload_factors": workload_factors,
        "proc_terms": proc_terms,
        "coord_cycles": float(coord_cycles),
        "base_proc": base_proc,
        "max_proc": max_proc,
        "coord_compute_time": float(coord_compute_time),
        "stage_total": float(max_proc + coord_compute_time),
    }


def split_activation_footprint(
    base_need: float,
    base_cycles: float,
    traffic_bits: float,
    base_cost: float,
    replica_count: int,
    split_delay_overhead_ratio: float,
    split_coord_delay_s: float,
    split_merge_delay_s: float,
    split_sync_traffic_ratio: float,
    split_consistency_delay_s: float,
    strict_full_replica: bool = False,
    strict_scale: float = 1.0,
    share_spec: Optional[str] = None,
) -> float:
    profile = build_split_stage_profile(
        base_need=base_need,
        base_cycles=base_cycles,
        replica_count=replica_count,
        split_delay_overhead_ratio=split_delay_overhead_ratio,
        split_coord_delay_s=split_coord_delay_s,
        split_merge_delay_s=split_merge_delay_s,
        split_consistency_delay_s=split_consistency_delay_s,
        strict_full_replica=strict_full_replica,
        strict_scale=strict_scale,
        share_spec=share_spec,
    )
    resource_increase = max(
        split_stage_total_resource(base_need, replica_count, strict_full_replica, strict_scale, share_spec) - float(base_need),
        0.0,
    )
    instantiation_proxy = float(max(replica_count - 1, 0)) * max(float(base_cost), 0.0)
    if strict_full_replica:
        delay_proxy = max(profile["stage_total"] - profile["base_proc"], 0.0)
        sync_proxy = float(max(replica_count - 1, 0)) * float(split_sync_traffic_ratio) * float(traffic_bits) / 1e6
        return float(resource_increase + instantiation_proxy + delay_proxy * 1e3 + sync_proxy)
    return float(resource_increase + instantiation_proxy + float(profile["coord_compute_time"]) * 1e3)



def compute_e2e_delay_breakdown(
    chain: List[int],
    primary_placement: dict,
    cpu_demand_row: np.ndarray,
    cycles_row: np.ndarray,
    access_delay_s: float,
    L: np.ndarray,
    BW: np.ndarray,
    traffic_bits: float,
    # --- VNF splitting / parallel replicas (optional) ---
    # split_mask_row[v]==1 indicates that VNF v is deployed as a split/parallelized component.
    #
    # NOTE:
    #   The main calibrated model uses workload-derived coordination/merge compute and
    #   placement-aware fan-out/fan-in paths. Legacy synchronization and coordination
    #   parameters are kept only for strict ablations.
    split_mask_row: Optional[np.ndarray] = None,
    split_replica_count: int = 2,
    # legacy strict-ablation parameters (ignored by the calibrated main model)
    split_delay_overhead_ratio: float = 0.15,
    split_coord_delay_s: float = 0.0,
    split_merge_delay_s: float = 0.0,
    split_consistency_delay_s: float = 0.0,
    split_sync_traffic_ratio: float = 0.0,
    strict_split_resource_ablation: bool = False,
    split_replica_resource_scale: float = 1.0,
    split_workload_shares_spec: Optional[str] = None,
) -> Dict[str, float]:
    """Compute E2E latency with a simple breakdown.

    total = access + processing + network

    network = Σ_hops [ propagation + transmission ]
           = Σ_hops [ L(ca,cb) + bits/BW(ca,cb) ]

    This breakdown is useful for derived metrics (avg/p95) and for addressing reviewers' concerns
    about missing coordination delays and synchronization traffic when VNF splitting is enabled.
    """
    access = float(access_delay_s)

    proc = 0.0
    proc_over_split = 0.0

    net = 0.0
    net_over_split = 0.0

    transmitted_bits = 0.0
    transmitted_bits_over_split = 0.0

    # processing (per VNF)
    for v in chain:
        v = int(v)
        cpu = float(cpu_demand_row[v])
        if cpu <= 0.0:
            return {
                'total': float('inf'),
                'access': access,
                'processing': float('inf'),
                'network': float('inf'),
                'split_processing_overhead': float('inf'),
                'split_network_overhead': float('inf'),
                'transmitted_bits': float('inf'),
                'split_transmitted_bits': float('inf'),
            }

        base_proc = float(cycles_row[v]) / (cpu * 1e6)

        if split_mask_row is not None and int(split_mask_row[v]) == 1 and int(split_replica_count) >= 2:
            profile = build_split_stage_profile(
                base_need=cpu,
                base_cycles=float(cycles_row[v]),
                replica_count=int(split_replica_count),
                split_delay_overhead_ratio=float(split_delay_overhead_ratio),
                split_coord_delay_s=float(split_coord_delay_s),
                split_merge_delay_s=float(split_merge_delay_s),
                split_consistency_delay_s=float(split_consistency_delay_s),
                strict_full_replica=bool(strict_split_resource_ablation),
                strict_scale=float(split_replica_resource_scale),
                share_spec=split_workload_shares_spec,
            )
            proc += float(profile["stage_total"])
            proc_over_split += float(max(profile["stage_total"] - base_proc, 0.0))
        else:
            proc += base_proc

    # network between consecutive VNFs
    for a, b in zip(chain[:-1], chain[1:]):
        a = int(a)
        b = int(b)
        ca = primary_placement.get(a, None)
        cb = primary_placement.get(b, None)
        if ca is None or cb is None:
            return {
                'total': float('inf'),
                'access': access,
                'processing': proc,
                'network': float('inf'),
                'split_processing_overhead': proc_over_split,
                'split_network_overhead': float('inf'),
                'transmitted_bits': transmitted_bits,
                'split_transmitted_bits': transmitted_bits_over_split,
            }

        if ca != cb:
            # propagation delay
            net += float(L[ca, cb])

            # transmission delay
            bw = float(BW[ca, cb])
            if not (bw > 0.0 and np.isfinite(bw)):
                continue

            bits_eff = float(traffic_bits)
            if bool(strict_split_resource_ablation) and split_mask_row is not None and int(split_replica_count) >= 2:
                if int(split_mask_row[a]) == 1 or int(split_mask_row[b]) == 1:
                    extra_bits = float(traffic_bits) * float(split_sync_traffic_ratio) * float(int(split_replica_count) - 1)
                    bits_eff += extra_bits
                    net_over_split += extra_bits / bw
                    transmitted_bits_over_split += extra_bits

            net += bits_eff / bw
            transmitted_bits += bits_eff

    total = access + proc + net
    return {
        'total': float(total),
        'access': float(access),
        'processing': float(proc),
        'network': float(net),
        'split_processing_overhead': float(proc_over_split),
        'split_network_overhead': float(net_over_split),
        'transmitted_bits': float(transmitted_bits),
        'split_transmitted_bits': float(transmitted_bits_over_split),
    }

def compute_e2e_delay_breakdown_forkjoin(
    chain: List[int],
    primary_placement: Dict[int, object],  # v -> int (non-split) OR v -> List[int] (split replicas)
    cpu_demand_row: np.ndarray,
    cycles_row: np.ndarray,
    access_delay_s: float,
    L: np.ndarray,
    BW: np.ndarray,
    traffic_bits: float,
    # --- explicit VNF splitting (fork-join) ---
    split_replica_count: int = 2,
    split_delay_overhead_ratio: float = 0.15,
    split_coord_delay_s: float = 0.0,
    split_merge_delay_s: float = 0.0,
    split_consistency_delay_s: float = 0.0,
    split_sync_traffic_ratio: float = 0.0,
    strict_split_resource_ablation: bool = False,
    split_replica_resource_scale: float = 1.0,
    split_workload_shares_spec: Optional[str] = None,
) -> Dict[str, float]:
    """
    Placement-aware fork-join model with explicit replica placements.

    For split stage l with replicas on cloudlets {k_{l,v}}:
      entry = anchor node of stage l (we use the first replica's cloudlet)
      next_entry = anchor node of stage l+1 (first replica of next stage)

      Main model:
        T_l = t_cm + max_v [ t(entry->k_{l,v}) + t_proc(v) + t(k_{l,v}->next_entry) ]
      where t_cm is workload-derived coordination/merge compute.

      Strict ablation:
        T_l keeps the legacy parameterized coordination / synchronization penalties.
    """

    access = float(access_delay_s)
    total = access

    processing = 0.0
    network = 0.0
    proc_over_split = 0.0
    net_over_split = 0.0

    transmitted_bits = 0.0
    split_transmitted_bits = 0.0

    if not chain:
        return {
            "total": total, "access": access,
            "processing": 0.0, "network": 0.0,
            "split_processing_overhead": 0.0, "split_network_overhead": 0.0,
            "transmitted_bits": 0.0, "split_transmitted_bits": 0.0,
            "split_overhead_ratio": 0.0,
        }

    # stage replicas in chain order
    stage_replicas: List[List[int]] = []
    stage_entry: List[int] = []
    for v in chain:
        p = primary_placement.get(int(v), None)
        if p is None:
            return {"total": float("inf"), "access": access,
                    "processing": float("inf"), "network": float("inf"),
                    "split_processing_overhead": float("inf"), "split_network_overhead": float("inf"),
                    "transmitted_bits": 0.0, "split_transmitted_bits": 0.0,
                    "split_overhead_ratio": 0.0}
        if isinstance(p, (list, tuple, np.ndarray)):
            nodes = [int(x) for x in list(p)]
        else:
            nodes = [int(p)]
        stage_replicas.append(nodes)
        stage_entry.append(nodes[0])

    def _net_delay_bits(src: int, dst: int, bits: float) -> Tuple[float, float]:
        if src == dst:
            return 0.0, 0.0
        bw = float(BW[src, dst])
        if bw <= 0 or not np.isfinite(bw):
            return float("inf"), 0.0
        return float(L[src, dst] + bits / bw), float(bits)

    for idx, v in enumerate(chain):
        v = int(v)
        nodes = stage_replicas[idx]
        entry = stage_entry[idx]
        next_entry = stage_entry[idx + 1] if idx < len(chain) - 1 else entry

        cpu_need = float(cpu_demand_row[v])
        base_proc = float(cycles_row[v] / (cpu_need * 1e6)) if cpu_need > 0 else 0.0

        if len(nodes) <= 1:
            # non-split: proc + hop to next stage anchor
            processing += base_proc
            hop = 0.0
            if idx < len(chain) - 1:
                hop, bits_sent = _net_delay_bits(nodes[0], next_entry, float(traffic_bits))
                network += hop
                transmitted_bits += bits_sent
            total += base_proc + hop
            continue

        V = int(len(nodes))
        profile = build_split_stage_profile(
            base_need=cpu_need,
            base_cycles=float(cycles_row[v]),
            replica_count=V,
            split_delay_overhead_ratio=float(split_delay_overhead_ratio),
            split_coord_delay_s=float(split_coord_delay_s),
            split_merge_delay_s=float(split_merge_delay_s),
            split_consistency_delay_s=float(split_consistency_delay_s),
            strict_full_replica=bool(strict_split_resource_ablation),
            strict_scale=float(split_replica_resource_scale),
            share_spec=split_workload_shares_spec,
        )
        proc_terms = np.asarray(profile["proc_terms"], dtype=float)
        shares = np.asarray(profile["shares"], dtype=float)
        coord_overhead = float(profile["coord_compute_time"])
        proc_over_split += float(max(profile["stage_total"] - base_proc, 0.0))

        best_path_time = -1.0
        best_in = best_out = best_proc = 0.0
        best_net_over = 0.0
        stage_bits_total = 0.0
        baseline_stage_bits = float(traffic_bits) if idx < len(chain) - 1 else 0.0

        for r_idx, c_rep in enumerate(nodes):
            if strict_split_resource_ablation:
                in_bits_model = float(traffic_bits)
                out_bits_model = float(traffic_bits) * (1.0 + float(split_sync_traffic_ratio) * (V - 1))
                extra_sync_bits = max(out_bits_model - float(traffic_bits), 0.0)
            else:
                in_bits_model = float(traffic_bits) * float(shares[r_idx])
                out_bits_model = float(traffic_bits) * float(shares[r_idx])
                extra_sync_bits = 0.0

            in_delay, in_bits = _net_delay_bits(entry, int(c_rep), in_bits_model)
            out_delay, out_bits = _net_delay_bits(int(c_rep), next_entry, out_bits_model)

            transmitted_bits += in_bits + out_bits
            stage_bits_total += in_bits + out_bits
            if strict_split_resource_ablation:
                split_transmitted_bits += max(extra_sync_bits, 0.0)

            net_over = 0.0
            if out_bits > 0 and extra_sync_bits > 0 and np.isfinite(float(BW[int(c_rep), next_entry])):
                net_over = float(extra_sync_bits / float(BW[int(c_rep), next_entry]))

            path_time = in_delay + float(proc_terms[r_idx]) + out_delay
            if path_time > best_path_time:
                best_path_time = path_time
                best_in, best_proc, best_out = in_delay, float(proc_terms[r_idx]), out_delay
                best_net_over = net_over

        processing += best_proc
        network += (best_in + best_out)
        net_over_split += best_net_over
        if not strict_split_resource_ablation:
            split_transmitted_bits += max(stage_bits_total - baseline_stage_bits, 0.0)

        total += best_path_time + coord_overhead

    split_overhead_ratio = float((proc_over_split + net_over_split) / total) if (total > 1e-12 and np.isfinite(total)) else 0.0

    return {
        "total": float(total),
        "access": float(access),
        "processing": float(processing),
        "network": float(network),
        "split_processing_overhead": float(proc_over_split),
        "split_network_overhead": float(net_over_split),
        "transmitted_bits": float(transmitted_bits),
        "split_transmitted_bits": float(split_transmitted_bits),
        "split_overhead_ratio": float(split_overhead_ratio),
    }


def compute_e2e_delay_seconds(
    chain: List[int],
    primary_placement: dict,
    cpu_demand_row: np.ndarray,
    cycles_row: np.ndarray,
    access_delay_s: float,
    L: np.ndarray,
    BW: np.ndarray,
    traffic_bits: float,
    # --- VNF splitting / parallel replicas (optional) ---
    split_mask_row: Optional[np.ndarray] = None,
    split_replica_count: int = 2,
    split_delay_overhead_ratio: float = 0.15,
    split_coord_delay_s: float = 0.0,
    split_merge_delay_s: float = 0.0,
    split_consistency_delay_s: float = 0.0,
    split_sync_traffic_ratio: float = 0.0,
    strict_split_resource_ablation: bool = False,
    split_replica_resource_scale: float = 1.0,
    split_workload_shares_spec: Optional[str] = None,
) -> float:
    """Backwards-compatible wrapper: returns total E2E delay only."""
    d = compute_e2e_delay_breakdown(
        chain=chain,
        primary_placement=primary_placement,
        cpu_demand_row=cpu_demand_row,
        cycles_row=cycles_row,
        access_delay_s=access_delay_s,
        L=L,
        BW=BW,
        traffic_bits=traffic_bits,
        split_mask_row=split_mask_row,
        split_replica_count=split_replica_count,
        split_delay_overhead_ratio=split_delay_overhead_ratio,
        split_coord_delay_s=split_coord_delay_s,
        split_merge_delay_s=split_merge_delay_s,
        split_consistency_delay_s=split_consistency_delay_s,
        split_sync_traffic_ratio=split_sync_traffic_ratio,
        strict_split_resource_ablation=strict_split_resource_ablation,
        split_replica_resource_scale=split_replica_resource_scale,
        split_workload_shares_spec=split_workload_shares_spec,
    )
    return float(d['total'])

# -----------------------------
# Data loading
# -----------------------------
def load_cloudlets(edge_csv_path: str, region: str) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(edge_csv_path)
    df_reg = df[df["Region"] == region].reset_index(drop=True)
    if df_reg.empty:
        raise ValueError(
            f"No cloudlets found for region={region}. "
            f"Try one of: {sorted(df['Region'].unique())[:20]} ..."
        )
    cpu_cores = df_reg["CPU (Core)"].to_numpy(dtype=float)
    return df_reg, cpu_cores


# -----------------------------
# Parameter generation (same logic as offline code)
# -----------------------------
def generate_time_reliability_params(
    rng: np.random.Generator,
    n: int,
    k: int,
    sfc: np.ndarray,
    computing_resources: np.ndarray,
):
    """Generate per-request QoS (latency) + reliability parameters.

    IMPORTANT: Compared with the previous (incorrect) model, we apply the UE access transmission delay
    ONCE per request (UE->ingress), and compute VNF processing delay only for VNFs that are actually
    in the chain (sfc==1). Inter-cloudlet propagation/transmission delays are computed AFTER placement.
    """
    # (1) Access (UE -> ingress cloudlet): wireless-like rate, applied once per request
    D_access = 30 + (30 - 10) * rng.random(n)  # [30,50] (abstract distance / pathloss factor)
    B = 5e6
    sigma2 = 1e-13
    P = 0.1
    theta = -4
    g = D_access ** theta
    W_access = B * np.log2(1 + (P * g) / sigma2)  # bps
    traffic_bits = (500 + (1000 - 500) * rng.random(n)) * 1000  # [5e5,1e6] bits
    access_delay = np.divide(traffic_bits, W_access, out=np.zeros_like(traffic_bits, dtype=float), where=(W_access != 0))

    # (2) VNF processing cycles (instructions), per request per VNF
    cycles = ((1 + 3 * rng.random((n, k))) * 1e6) * sfc  # only VNFs in chain have cycles

    # (3) Latency budget
    ST = (100 + (500 - 100) * rng.random(n)) / 1000  # [0.1,0.5] s

    # (4) Reliability
    vnf_reliability = 0.9 + 0.09 * rng.random((n, k))
    vnf_reliability[sfc == 0] = 0.0
    required_reliability = 0.9 + 0.09 * rng.random(n)

    # return 8 values to preserve call sites (names may be re-interpreted):
    #   t: access_delay, t3: cycles, D: D_access, W: W_access, N: traffic_bits
    return access_delay, cycles, D_access, W_access, traffic_bits, ST, vnf_reliability, required_reliability


def effective_vnf_reliability(r: float, backup_level: int, active_reliability: Optional[float] = None) -> float:
    active_reliability = float(r) if active_reliability is None else float(active_reliability)
    return stage_reliability_with_backups(active_reliability, float(r), int(backup_level))


def calc_reli_gain(
    r: float,
    backup_level: int,
    rel_main: float,
    active_reliability: Optional[float] = None,
    current_stage_reliability: Optional[float] = None,
) -> float:
    current_level = max(int(backup_level) - 1, 0)
    old_v = current_stage_reliability
    if old_v is None:
        old_v = effective_vnf_reliability(r, current_level, active_reliability=active_reliability)
    new_v = effective_vnf_reliability(r, int(backup_level), active_reliability=active_reliability)
    if old_v <= 0.0:
        return 0.0
    return float(rel_main) * new_v / old_v - float(rel_main)


def apply_next_backup(
    current_reli: float,
    r: float,
    current_backup_level: int,
    active_reliability: Optional[float] = None,
    current_stage_reliability: Optional[float] = None,
) -> float:
    old_v = current_stage_reliability
    if old_v is None:
        old_v = effective_vnf_reliability(r, current_backup_level, active_reliability=active_reliability)
    new_v = effective_vnf_reliability(r, current_backup_level + 1, active_reliability=active_reliability)
    if old_v <= 0.0:
        return current_reli
    return float(current_reli) * (new_v / old_v)


def select_backups_for_request(
    spec: AlgorithmSpec,
    primary_vnfs: np.ndarray,
    active_rel_row: np.ndarray,
    base_rel_row: np.ndarray,
    required_reliability: float,
    computing_row: np.ndarray,
    algo_rng: np.random.Generator,
    max_backup_level: int,
) -> Tuple[List[List[int]], float]:
    backup_level = {int(v): 0 for v in primary_vnfs}
    used_backups: List[List[int]] = []

    current_reli = request_reliability_with_backups(primary_vnfs, active_rel_row, base_rel_row, backup_level)
    if current_reli >= required_reliability or max_backup_level <= 0:
        return used_backups, current_reli

    def _candidate_update(v_id: int, cur_level: int) -> Optional[Tuple[float, float, float]]:
        if cur_level >= int(max_backup_level):
            return None
        cur_stage = stage_reliability_with_backups(
            active_reliability=float(active_rel_row[v_id]),
            backup_r=float(base_rel_row[v_id]),
            backup_count=cur_level,
        )
        new_stage = stage_reliability_with_backups(
            active_reliability=float(active_rel_row[v_id]),
            backup_r=float(base_rel_row[v_id]),
            backup_count=cur_level + 1,
        )
        if new_stage <= cur_stage + 1e-12:
            return None
        new_reli = current_reli / max(cur_stage, 1e-12) * new_stage
        gain = float(new_reli - current_reli)
        return cur_stage, new_stage, gain

    if spec.backup_policy == "gain_ratio":
        while current_reli < required_reliability:
            best_choice: Optional[Tuple[int, float, float, float]] = None
            for v in primary_vnfs:
                v_id = int(v)
                update = _candidate_update(v_id, int(backup_level[v_id]))
                if update is None:
                    continue
                cur_stage, new_stage, gain = update
                ratio = gain / max(float(computing_row[v_id]), 1e-6)
                if best_choice is None or ratio > best_choice[1]:
                    best_choice = (v_id, float(ratio), float(cur_stage), float(new_stage))
            if best_choice is None:
                break
            v_id, _ratio, cur_stage, new_stage = best_choice
            current_reli = current_reli / max(cur_stage, 1e-12) * new_stage
            backup_level[v_id] += 1
            used_backups.append([v_id, 1])
        return used_backups, float(current_reli)

    if spec.backup_policy == "resource_reliability_round_robin":
        ordered_vnfs = sorted(
            (int(v) for v in primary_vnfs),
            key=lambda v_id: float(computing_row[v_id]) * float(base_rel_row[v_id]),
            reverse=True,
        )
        progress = True
        while current_reli < required_reliability and progress:
            progress = False
            for v_id in ordered_vnfs:
                update = _candidate_update(v_id, int(backup_level[v_id]))
                if update is None:
                    continue
                cur_stage, new_stage, _gain = update
                current_reli = current_reli / max(cur_stage, 1e-12) * new_stage
                backup_level[v_id] += 1
                used_backups.append([v_id, 1])
                progress = True
                if current_reli >= required_reliability:
                    break
        return used_backups, float(current_reli)

    if spec.backup_policy == "low_reliability_first":
        ordered_vnfs = sorted((int(v) for v in primary_vnfs), key=lambda v_id: float(base_rel_row[v_id]))
    elif spec.backup_policy == "random_sequential":
        ordered_vnfs = [int(v) for v in primary_vnfs]
        algo_rng.shuffle(ordered_vnfs)
    else:
        raise ValueError(f"Unsupported backup policy: {spec.backup_policy}")

    for v_id in ordered_vnfs:
        while current_reli < required_reliability and backup_level[v_id] < int(max_backup_level):
            update = _candidate_update(v_id, int(backup_level[v_id]))
            if update is None:
                break
            cur_stage, new_stage, _gain = update
            current_reli = current_reli / max(cur_stage, 1e-12) * new_stage
            backup_level[v_id] += 1
            used_backups.append([v_id, 1])
        if current_reli >= required_reliability:
            break

    return used_backups, float(current_reli)


def select_vnf_instances_online(
    algo: str,
    n: int,
    k: int,
    sfc: np.ndarray,
    vnf_reliability: np.ndarray,
    required_reliability: np.ndarray,
    computing_resources: np.ndarray,
    pay: np.ndarray,
    cost_vnf: np.ndarray,
    cycles: np.ndarray,
    traffic_bits: np.ndarray,
    algo_rng: np.random.Generator,
    max_backup_level: int,
    diversity: bool = False,
    split_candidate_mask: Optional[np.ndarray] = None,
    split_replica_count: int = 2,
    split_delay_overhead_ratio: float = 0.15,
    split_coord_delay_s: float = 0.0,
    split_merge_delay_s: float = 0.0,
    split_sync_traffic_ratio: float = 0.0,
    split_consistency_delay_s: float = 0.0,
    strict_split_resource_ablation: bool = False,
    split_replica_resource_scale: float = 1.0,
    split_reliability_mode: str = "identical_active_pool",
    split_recovery_coverage: float = 1.0,
    split_workload_shares_spec: Optional[str] = None,
    a: float = 50,
):
    spec = get_algorithm_spec(algo)
    deploy_sets: List[Optional[List[List[int]]]] = [None] * n
    final_reli = np.zeros(n, dtype=float)
    accepted_mask = np.zeros(n, dtype=bool)
    configured_split_mask = np.zeros((n, k), dtype=int)
    qoe_val = 0.0

    for i_req in range(n):
        primary_vnfs = np.where(sfc[i_req, :] == 1)[0]
        if len(primary_vnfs) == 0:
            continue

        base_rel_row = np.asarray(vnf_reliability[i_req], dtype=float)
        active_rel_row = base_rel_row.copy()
        backup_level: Dict[int, int] = {int(v): 0 for v in primary_vnfs}
        current_reli = request_reliability_with_backups(primary_vnfs, active_rel_row, base_rel_row, backup_level)

        split_row = split_candidate_mask[i_req] if (diversity and split_candidate_mask is not None) else np.zeros(k, dtype=int)
        if diversity and int(split_replica_count) >= 2:
            while current_reli < required_reliability[i_req]:
                best_choice: Optional[Tuple[int, float, float, float]] = None
                for v in primary_vnfs:
                    v_id = int(v)
                    if int(split_row[v_id]) != 1 or int(configured_split_mask[i_req, v_id]) == 1:
                        continue
                    cur_stage = stage_reliability_with_backups(
                        active_reliability=float(active_rel_row[v_id]),
                        backup_r=float(base_rel_row[v_id]),
                        backup_count=0,
                    )
                    new_active = split_active_reliability(
                        float(base_rel_row[v_id]),
                        int(split_replica_count),
                        mode=str(split_reliability_mode),
                        recovery_coverage=float(split_recovery_coverage),
                    )
                    new_stage = stage_reliability_with_backups(
                        active_reliability=float(new_active),
                        backup_r=float(base_rel_row[v_id]),
                        backup_count=0,
                    )
                    if new_stage <= cur_stage + 1e-12:
                        continue
                    new_reli = current_reli / max(cur_stage, 1e-12) * new_stage
                    gain = float(new_reli - current_reli)
                    footprint = split_activation_footprint(
                        base_need=float(computing_resources[i_req, v_id]),
                        base_cycles=float(cycles[i_req, v_id]),
                        traffic_bits=float(traffic_bits[i_req]),
                        base_cost=float(cost_vnf[i_req, v_id]),
                        replica_count=int(split_replica_count),
                        split_delay_overhead_ratio=float(split_delay_overhead_ratio),
                        split_coord_delay_s=float(split_coord_delay_s),
                        split_merge_delay_s=float(split_merge_delay_s),
                        split_sync_traffic_ratio=float(split_sync_traffic_ratio),
                        split_consistency_delay_s=float(split_consistency_delay_s),
                        strict_full_replica=bool(strict_split_resource_ablation),
                        strict_scale=float(split_replica_resource_scale),
                        share_spec=split_workload_shares_spec,
                    )
                    score = gain / max(footprint, 1e-9)
                    if best_choice is None or score > best_choice[1]:
                        best_choice = (v_id, float(score), float(new_active), float(new_reli))
                if best_choice is None:
                    break
                v_id, _score, new_active, new_reli = best_choice
                configured_split_mask[i_req, v_id] = 1
                active_rel_row[v_id] = float(new_active)
                current_reli = float(new_reli)

        tmp_set = [[int(v), 0] for v in primary_vnfs]
        if current_reli < required_reliability[i_req]:
            used_backups, current_reli = select_backups_for_request(
                spec=spec,
                primary_vnfs=primary_vnfs,
                active_rel_row=active_rel_row,
                base_rel_row=base_rel_row,
                required_reliability=float(required_reliability[i_req]),
                computing_row=computing_resources[i_req],
                algo_rng=algo_rng,
                max_backup_level=int(max_backup_level),
            )
        else:
            used_backups = []

        final_reli[i_req] = float(current_reli)
        if current_reli >= required_reliability[i_req]:
            accepted_mask[i_req] = True
            deploy_sets[i_req] = tmp_set + used_backups
            qoe_val += np.exp(a * (current_reli - required_reliability[i_req]))
        else:
            deploy_sets[i_req] = []

    return deploy_sets, final_reli, accepted_mask, qoe_val, configured_split_mask

# -----------------------------
# Stage-2: Online placement (uses remaining capacity)
# -----------------------------

def mdca_assign_online(
    algo: str,
    deploy_sets: List[Optional[List[List[int]]]],
    accepted_mask: np.ndarray,
    computing_resources: np.ndarray,
    cap_left: np.ndarray,
    vct: np.ndarray,
    cost_vnf: np.ndarray,
    unit_comcost: np.ndarray,
    vnf_reliability: np.ndarray,
    req_reliability: np.ndarray,
    pay: np.ndarray,
    algo_rng: np.random.Generator,
    chains: List[List[int]],
    ST: np.ndarray,
    access_delay: np.ndarray,
    cycles: np.ndarray,
    traffic_bits: np.ndarray,
    L: np.ndarray,
    BW: np.ndarray,
    split_masks: Optional[np.ndarray] = None,
    split_replica_count: int = 2,
    explicit_split_replicas: bool = True,
    split_enforce_distinct: bool = False,
    strict_split_resource_ablation: bool = False,
    split_replica_resource_scale: float = 1.0,
    split_delay_overhead_ratio: float = 0.15,
    split_coord_delay_s: float = 0.0,
    split_merge_delay_s: float = 0.0,
    split_sync_traffic_ratio: float = 0.0,
    split_consistency_delay_s: float = 0.0,
    split_reliability_mode: str = "identical_active_pool",
    split_recovery_coverage: float = 1.0,
    req_global_ids: Optional[np.ndarray] = None,
    split_workload_shares_spec: Optional[str] = None,
    pvfp_policy: Optional[PvfpFdrlPlacementPolicy] = None,
):
    spec = get_algorithm_spec(algo)
    cap_left = cap_left.copy()
    idx_cloud_asc = np.argsort(vct)
    if spec.cloudlet_policy == "pvfp_fdrl" and pvfp_policy is None:
        pvfp_policy = PvfpFdrlPlacementPolicy(PvfpFdrlConfig())

    accepted_req_ids = np.where(accepted_mask)[0]
    if len(accepted_req_ids) == 0:
        return [], [], cap_left, 0.0, 0.0

    subsets: List[List[List[int]]] = []
    total_res_each: List[float] = []
    req_ids_kept: List[int] = []
    for i_req in accepted_req_ids:
        vnf_array = deploy_sets[i_req]
        if not vnf_array:
            continue

        split_row = split_masks[i_req] if (split_masks is not None) else None
        sum_res = 0.0
        for v_id, is_backup in vnf_array:
            v_id_i = int(v_id)
            is_bk_i = int(is_backup)
            base = float(computing_resources[i_req, v_id_i])
            if explicit_split_replicas and (split_row is not None) and is_bk_i == 0 and int(split_row[v_id_i]) == 1:
                sum_res += split_stage_total_resource(
                    base_need=base,
                    replica_count=int(split_replica_count),
                    strict_full_replica=bool(strict_split_resource_ablation),
                    strict_scale=float(split_replica_resource_scale),
                    share_spec=split_workload_shares_spec,
                )
            else:
                sum_res += base

        subsets.append(vnf_array)
        total_res_each.append(sum_res)
        req_ids_kept.append(int(i_req))

    if not subsets:
        return [], [], cap_left, 0.0, 0.0

    req_ids_arr = np.asarray(req_ids_kept, dtype=int)
    total_res_arr = np.asarray(total_res_each, dtype=float)
    if spec.request_order == "resource_asc":
        idx_sort = np.argsort(total_res_arr, kind="stable")
    elif spec.request_order == "reliability_asc":
        idx_sort = np.argsort(req_reliability[req_ids_arr], kind="stable")
    elif spec.request_order == "pay_desc":
        idx_sort = np.argsort(pay[req_ids_arr], kind="stable")[::-1]
    elif spec.request_order == "random":
        idx_sort = algo_rng.permutation(len(req_ids_arr))
    else:
        raise ValueError(f"Unsupported request order: {spec.request_order}")

    sorted_req_local = [req_ids_kept[i] for i in idx_sort]
    subsets_sorted = [subsets[i] for i in idx_sort]

    accepted_infos = []
    rejected_infos = []
    total_cost = 0.0
    total_cost_resource = 0.0

    for local_pos, i_req in enumerate(sorted_req_local):
        vnf_array = subsets_sorted[local_pos]

        if spec.instance_order == "resource_desc":
            vnf_res = [float(computing_resources[i_req, v_id]) for v_id, _ in vnf_array]
            order = np.argsort(vnf_res)[::-1]
            vnf_sorted = [vnf_array[i] for i in order]
        else:
            vnf_sorted = list(vnf_array)

        placed_instances = []
        local_cost = 0.0
        local_cost_res = 0.0
        feasible = True
        rid = int(req_global_ids[i_req]) if req_global_ids is not None else int(i_req)
        if spec.cloudlet_policy == "pvfp_fdrl" and pvfp_policy is not None:
            pvfp_policy.begin_request(rid)

        split_row = split_masks[i_req] if (split_masks is not None) else None
        expanded_instances: List[Tuple[int, int, int]] = []
        if explicit_split_replicas and (split_row is not None):
            for v_id, is_backup in vnf_sorted:
                v_id_i = int(v_id)
                is_bk_i = int(is_backup)
                if is_bk_i == 0 and int(split_row[v_id_i]) == 1:
                    for rep_idx in range(int(split_replica_count)):
                        expanded_instances.append((v_id_i, 0, int(rep_idx)))
                else:
                    expanded_instances.append((v_id_i, is_bk_i, -1))
        else:
            expanded_instances = [(int(v_id), int(is_backup), -1) for v_id, is_backup in vnf_sorted]

        used_cloudlets_for_split: Dict[int, set] = {}
        used_cloudlets_for_request: set = set()

        for v_id, is_backup, rep_idx in expanded_instances:
            base_need = float(computing_resources[i_req, v_id])
            if rep_idx >= 0:
                replica_needs = split_replica_resource_needs(
                    base_need=base_need,
                    replica_count=int(split_replica_count),
                    strict_full_replica=bool(strict_split_resource_ablation),
                    strict_scale=float(split_replica_resource_scale),
                    share_spec=split_workload_shares_spec,
                )
                res_need = float(replica_needs[rep_idx])
            else:
                res_need = float(base_need)
            cost_scale = (res_need / max(base_need, 1e-12)) if base_need > 0 else 1.0
            placed = False

            if rep_idx >= 0:
                pass_list = [True] if split_enforce_distinct else [True, False]
            else:
                pass_list = [False]
            for distinct_pass in pass_list:
                feasible_ids: List[int] = []
                if spec.cloudlet_policy in {"cheapest", "pvfp_fdrl"}:
                    iterator = idx_cloud_asc
                else:
                    iterator = algo_rng.permutation(len(cap_left))
                for c_id in iterator:
                    c_idx = int(c_id)
                    if spec.distinct_request_cloudlets and c_idx in used_cloudlets_for_request:
                        continue
                    if distinct_pass and (rep_idx >= 0):
                        used_split = used_cloudlets_for_split.setdefault(int(v_id), set())
                        if c_idx in used_split:
                            continue
                    if cap_left[c_idx] >= res_need:
                        feasible_ids.append(c_idx)
                        if spec.cloudlet_policy == "cheapest":
                            break

                if not feasible_ids:
                    continue

                if spec.cloudlet_policy == "random":
                    c_idx = int(algo_rng.choice(np.asarray(feasible_ids, dtype=int)))
                elif spec.cloudlet_policy == "pvfp_fdrl":
                    assert pvfp_policy is not None
                    selected = pvfp_policy.choose_cloudlet(
                        candidate_ids=feasible_ids,
                        cap_left=cap_left,
                        res_need=float(res_need),
                        base_need=float(base_need),
                        cost_row=np.asarray(unit_comcost[i_req, v_id], dtype=float) * float(cost_scale),
                        vct=vct,
                        access_delay_s=float(access_delay[i_req]),
                        latency_budget_s=float(ST[i_req]),
                        L=L,
                        BW=BW,
                        traffic_bits=float(traffic_bits[i_req]),
                        req_reliability=float(req_reliability[i_req]),
                        vnf_reliability=float(vnf_reliability[i_req, v_id]),
                        v_id=int(v_id),
                        is_backup=int(is_backup),
                        rep_idx=int(rep_idx),
                        chain=chains[int(i_req)],
                        placed_instances=placed_instances,
                    )
                    if selected is None:
                        continue
                    c_idx = int(selected)
                else:
                    c_idx = int(feasible_ids[0])

                cap_left[c_idx] -= res_need
                local_cost += float(cost_vnf[i_req, v_id] + unit_comcost[i_req, v_id, c_idx] * cost_scale)
                local_cost_res += float(unit_comcost[i_req, v_id, c_idx] * cost_scale)
                placed_instances.append((int(v_id), int(is_backup), int(c_idx), float(res_need), int(rep_idx)))
                if spec.distinct_request_cloudlets:
                    used_cloudlets_for_request.add(int(c_idx))
                if rep_idx >= 0:
                    used_cloudlets_for_split.setdefault(int(v_id), set()).add(int(c_idx))
                placed = True
                break

            if not placed:
                feasible = False
                break

        if feasible:
            primary_placement: Dict[int, object] = {}
            backup_counts: Dict[int, int] = {}
            for inst in placed_instances:
                v_id2 = int(inst[0])
                is_bk2 = int(inst[1])
                c_id2 = int(inst[2])
                rep_idx2 = int(inst[4])
                if is_bk2 != 0:
                    backup_counts[v_id2] = int(backup_counts.get(v_id2, 0)) + 1
                    continue
                if rep_idx2 >= 0:
                    cur = primary_placement.get(v_id2, None)
                    if not isinstance(cur, list):
                        cur = []
                        primary_placement[v_id2] = cur
                    cur.append(c_id2)
                elif v_id2 not in primary_placement:
                    primary_placement[v_id2] = c_id2

            chain = chains[int(i_req)]
            base_rel_row = np.asarray(vnf_reliability[i_req], dtype=float)
            realized_active_rel = base_rel_row.copy()
            if split_row is not None:
                for v in chain:
                    v_id = int(v)
                    if int(split_row[v_id]) != 1:
                        continue
                    nodes = primary_placement.get(v_id, None)
                    if isinstance(nodes, list) and len(nodes) >= 2 and len(set(nodes)) == len(nodes):
                        realized_active_rel[v_id] = split_active_reliability(
                            base_rel_row[v_id],
                            len(nodes),
                            mode=str(split_reliability_mode),
                            recovery_coverage=float(split_recovery_coverage),
                        )
                    else:
                        realized_active_rel[v_id] = float(base_rel_row[v_id])

            realized_reli = request_reliability_with_backups(
                primary_vnfs=np.asarray(chain, dtype=int),
                active_reliability_row=realized_active_rel,
                base_reliability_row=base_rel_row,
                backup_levels=backup_counts,
            )
            if realized_reli < float(req_reliability[i_req]):
                for inst in placed_instances:
                    cap_left[int(inst[2])] += float(inst[3])
                if spec.cloudlet_policy == "pvfp_fdrl" and pvfp_policy is not None:
                    pvfp_policy.finish_request(
                        rid,
                        {
                            "accepted": False,
                            "reason": "reliability",
                            "cost": float(local_cost),
                            "realized_reliability": float(realized_reli),
                        },
                    )
                rejected_infos.append(
                    {
                        "req_id": rid,
                        "reason": "reliability",
                        "local_req_index": int(i_req),
                        "realized_reliability": float(realized_reli),
                    }
                )
                continue

            if explicit_split_replicas:
                delay_detail = compute_e2e_delay_breakdown_forkjoin(
                    chain=chain,
                    primary_placement=primary_placement,
                    cpu_demand_row=computing_resources[i_req],
                    cycles_row=cycles[i_req],
                    access_delay_s=float(access_delay[i_req]),
                    L=L,
                    BW=BW,
                    traffic_bits=float(traffic_bits[i_req]),
                    split_replica_count=int(split_replica_count),
                    split_delay_overhead_ratio=float(split_delay_overhead_ratio),
                    split_coord_delay_s=float(split_coord_delay_s),
                    split_merge_delay_s=float(split_merge_delay_s),
                    split_consistency_delay_s=float(split_consistency_delay_s),
                    split_sync_traffic_ratio=float(split_sync_traffic_ratio),
                    strict_split_resource_ablation=bool(strict_split_resource_ablation),
                    split_replica_resource_scale=float(split_replica_resource_scale),
                    split_workload_shares_spec=split_workload_shares_spec,
                )
            else:
                delay_detail = compute_e2e_delay_breakdown(
                    chain=chain,
                    primary_placement=primary_placement,
                    cpu_demand_row=computing_resources[i_req],
                    cycles_row=cycles[i_req],
                    access_delay_s=float(access_delay[i_req]),
                    L=L,
                    BW=BW,
                    traffic_bits=float(traffic_bits[i_req]),
                    split_mask_row=(split_masks[i_req] if split_masks is not None else None),
                    split_replica_count=int(split_replica_count),
                    split_delay_overhead_ratio=float(split_delay_overhead_ratio),
                    split_coord_delay_s=float(split_coord_delay_s),
                    split_merge_delay_s=float(split_merge_delay_s),
                    split_sync_traffic_ratio=float(split_sync_traffic_ratio),
                    split_consistency_delay_s=float(split_consistency_delay_s),
                    strict_split_resource_ablation=bool(strict_split_resource_ablation),
                    split_replica_resource_scale=float(split_replica_resource_scale),
                    split_workload_shares_spec=split_workload_shares_spec,
                )

            e2e = float(delay_detail.get("total", float("inf")))
            if e2e > float(ST[i_req]):
                for inst in placed_instances:
                    cap_left[int(inst[2])] += float(inst[3])
                if spec.cloudlet_policy == "pvfp_fdrl" and pvfp_policy is not None:
                    pvfp_policy.finish_request(
                        rid,
                        {
                            "accepted": False,
                            "reason": "qos",
                            "cost": float(local_cost),
                            "e2e_delay": float(e2e),
                            "realized_reliability": float(realized_reli),
                        },
                    )
                rejected_infos.append(
                    {
                        "req_id": rid,
                        "reason": "qos",
                        "local_req_index": int(i_req),
                        "e2e_delay": float(e2e),
                        "realized_reliability": float(realized_reli),
                        "delay_detail": delay_detail,
                    }
                )
            else:
                accepted_infos.append(
                    {
                        "req_id": rid,
                        "instances": placed_instances,
                        "local_req_index": int(i_req),
                        "e2e_delay": float(e2e),
                        "realized_reliability": float(realized_reli),
                        "delay_detail": delay_detail,
                    }
                )
                total_cost += local_cost
                total_cost_resource += local_cost_res
                if spec.cloudlet_policy == "pvfp_fdrl" and pvfp_policy is not None:
                    pvfp_policy.finish_request(
                        rid,
                        {
                            "accepted": True,
                            "reason": "accepted",
                            "cost": float(local_cost),
                            "e2e_delay": float(e2e),
                            "realized_reliability": float(realized_reli),
                        },
                    )
        else:
            for inst in placed_instances:
                cap_left[int(inst[2])] += float(inst[3])
            if spec.cloudlet_policy == "pvfp_fdrl" and pvfp_policy is not None:
                pvfp_policy.finish_request(
                    rid,
                    {
                        "accepted": False,
                        "reason": "capacity",
                        "cost": float(local_cost),
                    },
                )
            rejected_infos.append({"req_id": rid, "reason": "capacity", "local_req_index": int(i_req)})

    return accepted_infos, rejected_infos, cap_left, total_cost, total_cost_resource


# -----------------------------
# Online simulation wrapper
# -----------------------------

def simulate_online(
    edge_csv_path: str,
    region: str,
    time_slots: int,
    arrival_rate: float,
    mean_service_time: float,
    k: int,
    seed: int,
    enable_departure: bool,
    diversity: bool,
    p_split: float,
    algo: str = "mdca",
    mechanism: Optional[str] = None,
    max_backup_level: Optional[int] = None,
    fixed_sfc_len: Optional[int] = None,
    fixed_reliability: Optional[float] = None,
    cap_base_min: int = 500,
    cap_base_max: int = 700,
    cap_scale: float = 1.0,
    split_replica_count: int = 2,
    split_delay_overhead_ratio: float = 0.15,
    split_coord_delay_s: float = 0.0,
    split_merge_delay_s: float = 0.0,
    split_sync_traffic_ratio: float = 0.0,
    split_consistency_delay_s: float = 0.0,
    split_reliability_mode: str = "identical_active_pool",
    split_recovery_coverage: float = 1.0,
    split_workload_shares_spec: Optional[str] = None,
    backhaul_knn_k: int = 4,
    backhaul_use_device_bw: bool = True,
    backhaul_bw_scale: float = 1.0,
    fiber_speed_mps: float = 2e8,
    switch_delay_s: float = 2e-4,
    explicit_split_replicas: bool = True,
    split_enforce_distinct: bool = False,
    strict_split_resource_ablation: bool = False,
    split_replica_resource_scale: float = 1.0,
    record_placement_trace: bool = False,
    pvfp_model_path: Optional[str] = None,
    pvfp_policy_mode: str = "eval",
    pvfp_fallback_if_missing: bool = True,
    pvfp_policy: Optional[PvfpFdrlPlacementPolicy] = None,
):
    spec = get_algorithm_spec(algo)
    if max_backup_level is None:
        max_backup_level = spec.default_max_backup_level

    trace_seed = derive_seed(
        "trace",
        edge_csv_path,
        region,
        time_slots,
        arrival_rate,
        mean_service_time,
        k,
        seed,
        fixed_sfc_len if fixed_sfc_len is not None else "none",
        fixed_reliability if fixed_reliability is not None else "none",
        cap_base_min,
        cap_base_max,
        cap_scale,
    )
    algo_seed = derive_seed(
        "algo",
        region,
        algo,
        seed,
        diversity,
        p_split,
        mechanism,
        max_backup_level,
        split_replica_count,
        split_delay_overhead_ratio,
        split_coord_delay_s,
        split_merge_delay_s,
        split_sync_traffic_ratio,
        split_consistency_delay_s,
            explicit_split_replicas,
            split_enforce_distinct,
            strict_split_resource_ablation,
            split_replica_resource_scale,
    )
    trace_rng = np.random.default_rng(trace_seed)
    algo_rng = np.random.default_rng(algo_seed)

    df_reg, cpu_cores = load_cloudlets(edge_csv_path, region)
    m = len(cpu_cores)

    cap_base = trace_rng.integers(cap_base_min, cap_base_max + 1, size=m)
    cap_total = (cpu_cores * cap_base * cap_scale).astype(float)
    cap_left = cap_total.copy()

    L_prop, BW_backhaul = build_latency_and_bandwidth(
        trace_rng,
        m,
        df_reg,
        fiber_speed_mps=fiber_speed_mps,
        switch_delay_s=switch_delay_s,
        knn_k=backhaul_knn_k,
        use_device_bandwidth=backhaul_use_device_bw,
        device_bw_scale=backhaul_bw_scale,
    )

    vct = (trace_rng.random(m) * (5 - 1) + 1) / 100

    if spec.cloudlet_policy == "pvfp_fdrl" and pvfp_policy is None:
        pvfp_policy = PvfpFdrlPlacementPolicy(
            PvfpFdrlConfig(
                checkpoint_path=pvfp_model_path,
                mode=str(pvfp_policy_mode),
                seed=int(algo_seed),
                fallback_if_missing=bool(pvfp_fallback_if_missing),
            )
        )

    trace_hasher = hashlib.sha256()
    update_trace_hash(
        trace_hasher,
        -1,
        {
            "cap_total": cap_total,
            "vct": vct,
            "latency": L_prop,
            "bandwidth": BW_backhaul,
        },
    )

    active_tasks = []
    next_req_id = 0

    total_arrivals = 0
    total_accepted = 0
    total_rejected = 0
    rga_accepted_total = 0
    rga_rejected_total = 0
    mdca_rejected_total = 0
    mdca_capacity_rejected_total = 0
    mdca_qos_rejected_total = 0
    mdca_reliability_rejected_total = 0

    total_cost = 0.0
    total_cost_resource = 0.0

    slot_log: List[Dict] = []
    placement_trace: List[Dict] = []
    arrivals_per_slot: List[int] = []

    all_accepted_delays: List[float] = []
    all_accepted_access: List[float] = []
    all_accepted_processing: List[float] = []
    all_accepted_network: List[float] = []
    all_accepted_split_proc_over: List[float] = []
    all_accepted_split_net_over: List[float] = []
    all_accepted_transmitted_bits: List[float] = []
    all_accepted_split_transmitted_bits: List[float] = []
    all_accepted_split_overhead_ratio: List[float] = []

    for t_slot in range(time_slots):
        if enable_departure and active_tasks:
            still_active = []
            for task in active_tasks:
                if task["end_slot"] <= t_slot:
                    for c_id, res in task["allocations"]:
                        cap_left[c_id] += res
                else:
                    still_active.append(task)
            active_tasks = still_active

        n_arrivals = int(trace_rng.poisson(arrival_rate))
        arrivals_per_slot.append(int(n_arrivals))

        if n_arrivals <= 0:
            update_trace_hash(trace_hasher, t_slot, {"arrivals": np.asarray([0], dtype=np.int32)})
            slot_log.append(
                {
                    "slot": t_slot,
                    "arrivals": 0,
                    "rga_accepted": 0,
                    "rga_rejected": 0,
                    "accepted": 0,
                    "rejected": 0,
                    "mdca_rejected": 0,
                    "capacity_rejected": 0,
                    "qos_rejected": 0,
                    "reliability_rejected": 0,
                    "avg_e2e_delay": 0.0,
                    "p95_e2e_delay": 0.0,
                    "avg_access_delay": 0.0,
                    "avg_processing_delay": 0.0,
                    "avg_network_delay": 0.0,
                    "avg_split_proc_overhead": 0.0,
                    "avg_split_net_overhead": 0.0,
                    "avg_transmitted_bits": 0.0,
                    "avg_split_transmitted_bits": 0.0,
                    "avg_split_overhead_ratio": 0.0,
                    "leftover_sum": float(np.sum(cap_left)),
                    "utilization": float(1.0 - (np.sum(cap_left) / np.sum(cap_total))) if np.sum(cap_total) > 0 else 0.0,
                    "active_tasks": int(len(active_tasks)),
                }
            )
            continue

        total_arrivals += n_arrivals
        req_global_ids = np.arange(next_req_id, next_req_id + n_arrivals, dtype=int)
        next_req_id += n_arrivals

        sfc, chains = generate_sfc_batch(trace_rng, n_arrivals, k, fixed_sfc_len=fixed_sfc_len)
        computing_resources = trace_rng.integers(40, 401, size=(n_arrivals, k)) * sfc

        t_access, cycles, D_access, W_access, traffic_bits, ST, vnf_reli, req_reli = generate_time_reliability_params(
            trace_rng, n_arrivals, k, sfc, computing_resources
        )
        if fixed_reliability is not None:
            req_reli = np.full(n_arrivals, float(fixed_reliability), dtype=float)

        pay = 10 + 20 * trace_rng.random(n_arrivals)
        service_times = np.floor(np.maximum(1.0, trace_rng.exponential(mean_service_time, size=n_arrivals))).astype(int)

        cost_vnf = trace_rng.integers(1, 4, size=(n_arrivals, k)) * sfc
        unit_comcost = computing_resources[:, :, None] * vct[None, None, :]

        chain_lengths = np.asarray([len(chain) for chain in chains], dtype=np.int16)
        flat_chains = np.asarray([v for chain in chains for v in chain], dtype=np.int16)
        update_trace_hash(
            trace_hasher,
            t_slot,
            {
                "arrivals": np.asarray([n_arrivals], dtype=np.int32),
                "chain_lengths": chain_lengths,
                "flat_chains": flat_chains,
                "sfc": sfc.astype(np.int8),
                "computing_resources": computing_resources.astype(np.int16),
                "access_delay": t_access.astype(np.float64),
                "cycles": cycles.astype(np.float64),
                "traffic_bits": traffic_bits.astype(np.float64),
                "latency_budget": ST.astype(np.float64),
                "vnf_reliability": vnf_reli.astype(np.float64),
                "required_reliability": req_reli.astype(np.float64),
                "pay": pay.astype(np.float64),
                "service_times": service_times.astype(np.int16),
                "cost_vnf": cost_vnf.astype(np.float64),
            },
        )

        split_candidate_mask = np.zeros((n_arrivals, k), dtype=int)
        if diversity:
            split_candidate_mask = (sfc == 1).astype(int)
            if float(p_split) < 1.0 - 1e-12:
                split_candidate_mask = (
                    (algo_rng.random((n_arrivals, k)) < float(p_split)) & (sfc == 1)
                ).astype(int)

        deploy_sets, final_reli, accepted_mask, _, configured_split_mask = select_vnf_instances_online(
            algo=algo,
            n=n_arrivals,
            k=k,
            sfc=sfc,
            vnf_reliability=vnf_reli,
            required_reliability=req_reli,
            computing_resources=computing_resources,
            pay=pay,
            cost_vnf=cost_vnf,
            cycles=cycles,
            traffic_bits=traffic_bits,
            algo_rng=algo_rng,
            max_backup_level=int(max_backup_level),
            diversity=bool(diversity),
            split_candidate_mask=split_candidate_mask,
            split_replica_count=int(split_replica_count),
            split_delay_overhead_ratio=float(split_delay_overhead_ratio),
            split_coord_delay_s=float(split_coord_delay_s),
            split_merge_delay_s=float(split_merge_delay_s),
            split_sync_traffic_ratio=float(split_sync_traffic_ratio),
            split_consistency_delay_s=float(split_consistency_delay_s),
            strict_split_resource_ablation=bool(strict_split_resource_ablation),
            split_replica_resource_scale=float(split_replica_resource_scale),
            split_reliability_mode=str(split_reliability_mode),
            split_recovery_coverage=float(split_recovery_coverage),
            split_workload_shares_spec=split_workload_shares_spec,
        )

        accepted_infos, rejected_infos, cap_left, batch_cost, batch_cost_res = mdca_assign_online(
            algo=algo,
            deploy_sets=deploy_sets,
            accepted_mask=accepted_mask,
            computing_resources=computing_resources,
            cap_left=cap_left,
            vct=vct,
            cost_vnf=cost_vnf,
            unit_comcost=unit_comcost,
            vnf_reliability=vnf_reli,
            req_reliability=req_reli,
            pay=pay,
            algo_rng=algo_rng,
            chains=chains,
            ST=ST,
            access_delay=t_access,
            cycles=cycles,
            traffic_bits=traffic_bits,
            L=L_prop,
            BW=BW_backhaul,
            split_masks=configured_split_mask if diversity else None,
            split_replica_count=int(split_replica_count),
            split_delay_overhead_ratio=float(split_delay_overhead_ratio),
            split_coord_delay_s=float(split_coord_delay_s),
            split_merge_delay_s=float(split_merge_delay_s),
            split_sync_traffic_ratio=float(split_sync_traffic_ratio),
            split_consistency_delay_s=float(split_consistency_delay_s),
            split_reliability_mode=str(split_reliability_mode),
            split_recovery_coverage=float(split_recovery_coverage),
            req_global_ids=req_global_ids,
            split_workload_shares_spec=split_workload_shares_spec,
            explicit_split_replicas=explicit_split_replicas,
            split_enforce_distinct=split_enforce_distinct,
            strict_split_resource_ablation=bool(strict_split_resource_ablation),
            split_replica_resource_scale=split_replica_resource_scale,
            pvfp_policy=pvfp_policy,
        )

        accepted_n = len(accepted_infos)

        slot_delays = [float(info.get("e2e_delay", 0.0)) for info in accepted_infos]
        slot_access = [float(info.get("delay_detail", {}).get("access", 0.0)) for info in accepted_infos]
        slot_processing = [float(info.get("delay_detail", {}).get("processing", 0.0)) for info in accepted_infos]
        slot_network = [float(info.get("delay_detail", {}).get("network", 0.0)) for info in accepted_infos]
        slot_split_proc_over = [float(info.get("delay_detail", {}).get("split_processing_overhead", 0.0)) for info in accepted_infos]
        slot_split_net_over = [float(info.get("delay_detail", {}).get("split_network_overhead", 0.0)) for info in accepted_infos]
        slot_tx_bits = [float(info.get("delay_detail", {}).get("transmitted_bits", 0.0)) for info in accepted_infos]
        slot_split_tx_bits = [float(info.get("delay_detail", {}).get("split_transmitted_bits", 0.0)) for info in accepted_infos]

        if slot_delays:
            all_accepted_delays.extend(slot_delays)
            all_accepted_access.extend(slot_access)
            all_accepted_processing.extend(slot_processing)
            all_accepted_network.extend(slot_network)
            all_accepted_split_proc_over.extend(slot_split_proc_over)
            all_accepted_split_net_over.extend(slot_split_net_over)
            all_accepted_transmitted_bits.extend(slot_tx_bits)
            all_accepted_split_transmitted_bits.extend(slot_split_tx_bits)
            for total_delay, proc_over, net_over in zip(slot_delays, slot_split_proc_over, slot_split_net_over):
                if total_delay > 0:
                    all_accepted_split_overhead_ratio.append(float((proc_over + net_over) / total_delay))

        if record_placement_trace and accepted_infos:
            for info in accepted_infos:
                local_idx = int(info["local_req_index"])
                chain = [int(v) for v in chains[local_idx]]
                split_row = configured_split_mask[local_idx] if diversity else np.zeros(k, dtype=int)
                active_first_stage = [
                    inst for inst in info.get("instances", [])
                    if int(inst[0]) == int(chain[0]) and int(inst[1]) == 0
                ]
                active_first_stage = sorted(active_first_stage, key=lambda inst: int(inst[4]))
                entry_cloudlet = int(active_first_stage[0][2]) if active_first_stage else None
                placement_trace.append(
                    {
                        "slot": int(t_slot),
                        "req_id": int(info["req_id"]),
                        "local_req_index": int(local_idx),
                        "chain": chain,
                        "split_mask": [int(split_row[int(v)]) for v in chain],
                        "entry_cloudlet": entry_cloudlet,
                        "latency_budget": float(ST[local_idx]),
                        "access_delay": float(t_access[local_idx]),
                        "access_distance_proxy": float(D_access[local_idx]),
                        "access_rate_bps": float(W_access[local_idx]),
                        "traffic_bits": float(traffic_bits[local_idx]),
                        "required_reliability": float(req_reli[local_idx]),
                        "realized_reliability": float(info.get("realized_reliability", 0.0)),
                        "e2e_delay": float(info.get("e2e_delay", 0.0)),
                        "delay_detail": {
                            str(k_detail): float(v_detail)
                            for k_detail, v_detail in dict(info.get("delay_detail", {})).items()
                        },
                        "vnf_base_reliability": {str(int(v)): float(vnf_reli[local_idx, int(v)]) for v in chain},
                        "instances": [
                            {
                                "vnf": int(inst[0]),
                                "is_backup": int(inst[1]),
                                "cloudlet": int(inst[2]),
                                "resource": float(inst[3]),
                                "replica": int(inst[4]),
                            }
                            for inst in info.get("instances", [])
                        ],
                    }
                )

        slot_avg_delay = float(np.mean(slot_delays)) if slot_delays else 0.0
        slot_p95_delay = float(np.percentile(slot_delays, 95)) if slot_delays else 0.0
        slot_avg_access = float(np.mean(slot_access)) if slot_access else 0.0
        slot_avg_processing = float(np.mean(slot_processing)) if slot_processing else 0.0
        slot_avg_network = float(np.mean(slot_network)) if slot_network else 0.0
        slot_avg_split_proc_over = float(np.mean(slot_split_proc_over)) if slot_split_proc_over else 0.0
        slot_avg_split_net_over = float(np.mean(slot_split_net_over)) if slot_split_net_over else 0.0
        slot_avg_tx_bits = float(np.mean(slot_tx_bits)) if slot_tx_bits else 0.0

        rejected_n = n_arrivals - accepted_n
        qos_rejected_n = int(sum(1 for info in rejected_infos if info.get("reason") == "qos")) if isinstance(rejected_infos, list) else 0
        reliability_rejected_n = int(sum(1 for info in rejected_infos if info.get("reason") == "reliability")) if isinstance(rejected_infos, list) else 0

        rga_accepted_n = int(np.sum(accepted_mask))
        rga_rejected_n = int(n_arrivals - rga_accepted_n)
        mdca_rejected_n = int(len(rejected_infos)) if isinstance(rejected_infos, list) else 0
        mdca_capacity_rejected_n = int(sum(1 for info in rejected_infos if info.get("reason") == "capacity")) if isinstance(rejected_infos, list) else 0
        mdca_qos_rejected_n = int(qos_rejected_n)
        mdca_reliability_rejected_n = int(reliability_rejected_n)

        rga_accepted_total += rga_accepted_n
        rga_rejected_total += rga_rejected_n
        mdca_rejected_total += mdca_rejected_n
        mdca_capacity_rejected_total += mdca_capacity_rejected_n
        mdca_qos_rejected_total += mdca_qos_rejected_n
        mdca_reliability_rejected_total += mdca_reliability_rejected_n

        total_accepted += accepted_n
        total_rejected += rejected_n
        total_cost += batch_cost
        total_cost_resource += batch_cost_res

        if enable_departure:
            for info in accepted_infos:
                local_idx = int(info["local_req_index"])
                dur = int(max(1, service_times[local_idx]))
                end_slot = t_slot + dur
                allocations = [(int(inst[2]), float(inst[3])) for inst in info["instances"]]
                active_tasks.append({"end_slot": end_slot, "allocations": allocations, "req_id": info["req_id"]})

        slot_log.append(
            {
                "slot": t_slot,
                "arrivals": int(n_arrivals),
                "rga_accepted": int(rga_accepted_n),
                "rga_rejected": int(rga_rejected_n),
                "mdca_rejected": int(mdca_rejected_n),
                "capacity_rejected": int(mdca_capacity_rejected_n),
                "reliability_rejected": int(mdca_reliability_rejected_n),
                "accepted": int(accepted_n),
                "rejected": int(rejected_n),
                "qos_rejected": int(qos_rejected_n),
                "avg_e2e_delay": float(slot_avg_delay),
                "p95_e2e_delay": float(slot_p95_delay),
                "avg_access_delay": float(slot_avg_access),
                "avg_processing_delay": float(slot_avg_processing),
                "avg_network_delay": float(slot_avg_network),
                "avg_split_proc_overhead": float(slot_avg_split_proc_over),
                "avg_split_net_overhead": float(slot_avg_split_net_over),
                "avg_transmitted_bits": float(slot_avg_tx_bits),
                "avg_split_transmitted_bits": float(np.mean(slot_split_tx_bits)) if slot_split_tx_bits else 0.0,
                "avg_split_overhead_ratio": float(np.mean([((proc_over + net_over) / total_delay) for total_delay, proc_over, net_over in zip(slot_delays, slot_split_proc_over, slot_split_net_over) if total_delay > 0])) if slot_delays else 0.0,
                "leftover_sum": float(np.sum(cap_left)),
                "utilization": float(1.0 - (np.sum(cap_left) / np.sum(cap_total))) if np.sum(cap_total) > 0 else 0.0,
                "active_tasks": int(len(active_tasks)),
            }
        )

    metrics = {
        "algo": algo,
        "region": region,
        "time_slots": time_slots,
        "arrival_rate": arrival_rate,
        "mean_service_time": mean_service_time,
        "seed": seed,
        "trace_seed": int(trace_seed),
        "algo_seed": int(algo_seed),
        "trace_hash": trace_hasher.hexdigest(),
        "arrivals_per_slot": arrivals_per_slot,
        "diversity": diversity,
        "p_split": p_split if diversity else None,
        "split_reliability_mode": str(split_reliability_mode) if diversity else None,
        "split_recovery_coverage": float(split_recovery_coverage) if diversity else None,
        "split_workload_shares": (",".join(f"{x:.6g}" for x in split_workload_shares(int(split_replica_count), split_workload_shares_spec)) if diversity else None),
        "mechanism": mechanism,
        "fixed_sfc_len": int(fixed_sfc_len) if fixed_sfc_len is not None else None,
        "fixed_reliability": float(fixed_reliability) if fixed_reliability is not None else None,
        "max_backup_level": int(max_backup_level),
        "explicit_split_replicas": bool(explicit_split_replicas),
        "split_enforce_distinct": bool(split_enforce_distinct),
        "strict_split_resource_ablation": bool(strict_split_resource_ablation),
        "split_replica_resource_scale": float(split_replica_resource_scale),
        "backhaul_knn_k": int(backhaul_knn_k),
        "backhaul_use_device_bw": bool(backhaul_use_device_bw),
        "backhaul_bw_scale": float(backhaul_bw_scale),
        "fiber_speed_mps": float(fiber_speed_mps),
        "switch_delay_s": float(switch_delay_s),
        "region_cloudlet_count": int(m),
        "cap_base_min": int(cap_base_min),
        "cap_base_max": int(cap_base_max),
        "cap_scale": float(cap_scale),
        "total_arrivals": total_arrivals,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "acceptance_ratio": (total_accepted / total_arrivals) if total_arrivals > 0 else 0.0,
        "avg_cost_per_arrival": (total_cost / total_arrivals) if total_arrivals > 0 else 0.0,
        "total_cost": total_cost,
        "total_cost_resource": total_cost_resource,
        "avg_cost_per_accepted": (total_cost / total_accepted) if total_accepted > 0 else 0.0,
        "final_utilization": float(1.0 - (np.sum(cap_left) / np.sum(cap_total))) if np.sum(cap_total) > 0 else 0.0,
        "avg_utilization_over_time": float(pd.DataFrame(slot_log)["utilization"].mean()) if len(slot_log) > 0 else 0.0,
        "avg_leftover_capacity_sum_over_time": float(pd.DataFrame(slot_log)["leftover_sum"].mean()) if len(slot_log) > 0 else 0.0,
        "final_leftover_capacity_sum": float(np.sum(cap_left)),
        "final_leftover_capacity_min": float(np.min(cap_left)),
        "final_leftover_capacity_max": float(np.max(cap_left)),
        "final_leftover_capacity_avg": float(np.mean(cap_left)),
        "avg_e2e_delay_accepted": float(np.mean(all_accepted_delays)) if all_accepted_delays else 0.0,
        "p95_e2e_delay_accepted": float(np.percentile(all_accepted_delays, 95)) if all_accepted_delays else 0.0,
        "avg_access_delay_accepted": float(np.mean(all_accepted_access)) if all_accepted_access else 0.0,
        "avg_processing_delay_accepted": float(np.mean(all_accepted_processing)) if all_accepted_processing else 0.0,
        "avg_network_delay_accepted": float(np.mean(all_accepted_network)) if all_accepted_network else 0.0,
        "avg_split_proc_overhead_accepted": float(np.mean(all_accepted_split_proc_over)) if all_accepted_split_proc_over else 0.0,
        "avg_split_net_overhead_accepted": float(np.mean(all_accepted_split_net_over)) if all_accepted_split_net_over else 0.0,
        "total_transmitted_bits_accepted": float(np.sum(all_accepted_transmitted_bits)) if all_accepted_transmitted_bits else 0.0,
        "avg_transmitted_bits_per_accepted": float(np.mean(all_accepted_transmitted_bits)) if all_accepted_transmitted_bits else 0.0,
        "total_split_transmitted_bits_accepted": float(np.sum(all_accepted_split_transmitted_bits)) if all_accepted_split_transmitted_bits else 0.0,
        "avg_split_transmitted_bits_per_accepted": float(np.mean(all_accepted_split_transmitted_bits)) if all_accepted_split_transmitted_bits else 0.0,
        "avg_split_overhead_ratio_accepted": float(np.mean(all_accepted_split_overhead_ratio)) if all_accepted_split_overhead_ratio else 0.0,
        "p95_split_overhead_ratio_accepted": float(np.percentile(all_accepted_split_overhead_ratio, 95)) if all_accepted_split_overhead_ratio else 0.0,
        "rga_accepted_total": int(rga_accepted_total),
        "rga_rejected_total": int(rga_rejected_total),
        "rga_rejection_ratio": (float(rga_rejected_total) / total_arrivals) if total_arrivals > 0 else 0.0,
        "mdca_rejected_total": int(mdca_rejected_total),
        "mdca_capacity_rejected_total": int(mdca_capacity_rejected_total),
        "mdca_reliability_rejected_total": int(mdca_reliability_rejected_total),
        "mdca_qos_rejected_total": int(mdca_qos_rejected_total),
        "mdca_rejection_ratio": (float(mdca_rejected_total) / total_arrivals) if total_arrivals > 0 else 0.0,
        "capacity_rejection_ratio": (float(mdca_capacity_rejected_total) / total_arrivals) if total_arrivals > 0 else 0.0,
        "reliability_rejection_ratio": (float(mdca_reliability_rejected_total) / total_arrivals) if total_arrivals > 0 else 0.0,
        "qos_rejected_total": int(mdca_qos_rejected_total),
        "qos_rejection_ratio": (float(mdca_qos_rejected_total) / total_arrivals) if total_arrivals > 0 else 0.0,
    }
    if spec.cloudlet_policy == "pvfp_fdrl":
        pvfp_status = pvfp_policy.status() if pvfp_policy is not None else {}
        metrics.update(
            {
                "pvfp_model_path": pvfp_model_path,
                "pvfp_policy_mode": str(pvfp_policy_mode),
                "pvfp_checkpoint_loaded": bool(pvfp_status.get("loaded_checkpoint", False)),
                "pvfp_fallback_active": bool(pvfp_status.get("fallback_active", False)),
                "pvfp_fallback_used": bool(pvfp_status.get("fallback_used", False)),
                "pvfp_backend": pvfp_status.get("backend", None),
                "pvfp_uses_pvfp_code": bool(pvfp_status.get("uses_pvfp_code", False)),
                "pvfp_source_path": pvfp_status.get("pvfp_source_path", None),
                "pvfp_training_steps": int(pvfp_status.get("training_steps", 0)),
                "pvfp_aggregation_rounds": int(pvfp_status.get("aggregation_rounds", 0)),
            }
        )
    if record_placement_trace:
        metrics["_placement_trace"] = placement_trace
    return metrics, slot_log



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_csv", type=str, default="Edge_devices.csv", help="Path to Edge_devices.csv")
    parser.add_argument("--region", type=str, default="Alabama", help="Region name, e.g., Alabama / Arizona")
    parser.add_argument("--time_slots", type=int, default=60, help="Number of time slots in the simulation horizon")
    parser.add_argument("--arrival_rate", type=float, default=10.0, help="Average arrivals per slot (Poisson)")
    parser.add_argument("--mean_service_time", type=float, default=5.0, help="Mean service time in slots (Exponential)")
    parser.add_argument("--k", type=int, default=6, help="Number of VNF types")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--algo", type=str, default="mdca", choices=["mdca", "dtsp", "hspa", "random", "pvfp_fdrl"], help="Online algorithm to run")
    parser.add_argument("--no_departure", action="store_true", help="Disable departures (requests never leave)")
    parser.add_argument("--diversity", action="store_true", help="Enable split-priority hybrid DMDCA with active-diversity replicas.")
    parser.add_argument(
        "--p_split",
        type=float,
        default=0.3,
        help="Stage-level split-eligibility sampling probability when diversity is enabled. Main experiments use 0.3; setting 1.0 makes all chain stages eligible.",
    )
    parser.add_argument("--fixed_sfc_len", type=int, default=None, help="Fix every request to the same SFC length (1..k)")
    parser.add_argument("--fixed_reliability", type=float, default=None, help="Fix every request to the same end-to-end reliability demand")
    parser.add_argument(
        "--mechanism",
        type=str,
        default=None,
        choices=["primary_only", "redundancy_only", "split_only", "dual"],
        help="Select an ablation mechanism preset. Overrides --diversity/--p_split/--max_backup_level accordingly.",
    )
    parser.add_argument(
        "--max_backup_level",
        type=int,
        default=None,
        help="Maximum backup level per VNF. If omitted, the algorithm-specific default is used.",
    )
    parser.add_argument("--cap_base_min", type=int, default=500, help="Random capacity base lower bound (before multiplying cpu_cores).")
    parser.add_argument("--cap_base_max", type=int, default=700, help="Random capacity base upper bound (before multiplying cpu_cores).")
    parser.add_argument("--cap_scale", type=float, default=1.0, help="Scale factor applied to total capacities (use <1.0 to create overload regimes).")
    parser.add_argument("--explicit_split_replicas", action="store_true", help="Explicitly deploy active replicas for each split VNF and evaluate placement-aware fork-join delay.")
    parser.add_argument("--split_enforce_distinct", action="store_true", help="Strict ablation: require split replicas of the same VNF to be placed on distinct cloudlets.")
    parser.add_argument("--strict_split_resource_ablation", action="store_true", help="Strict ablation: each split replica keeps full-workload semantics and uses --split_replica_resource_scale of the base resource.")
    parser.add_argument("--split_replica_resource_scale", type=float, default=1.0, help="Per-replica resource scale used only when --strict_split_resource_ablation is enabled.")
    parser.add_argument("--split_replica_count", type=int, default=2, help="Number of active replicas for a split VNF.")
    parser.add_argument("--split_workload_shares", type=str, default=None, help="Comma-separated workload shares for split replicas, e.g., 0.5,0.5 or 0.7,0.3. Values are normalized; default is equal sharing.")
    parser.add_argument(
        "--split_reliability_mode",
        type=str,
        default="identical_active_pool",
        choices=["identical_active_pool", "heterogeneous_upper_bound"],
        help="Reliability model for split active replicas. Main experiments use identical_active_pool; heterogeneous_upper_bound is ablation-only.",
    )
    parser.add_argument(
        "--split_recovery_coverage",
        type=float,
        default=1.0,
        help="Recovery coverage eta for split active replicas. eta=1 recovers the ideal active-pool form; eta=0 requires all active replicas to survive.",
    )
    parser.add_argument("--split_delay_overhead_ratio", type=float, default=0.15, help="Legacy strict-ablation knob: processing-side coordination overhead ratio per extra replica.")
    parser.add_argument("--split_coord_delay_s", type=float, default=0.0, help="Legacy strict-ablation knob: extra coordination delay per extra replica (seconds).")
    parser.add_argument("--split_merge_delay_s", type=float, default=0.0, help="Legacy strict-ablation knob: extra merge/aggregation delay (seconds).")
    parser.add_argument("--split_sync_traffic_ratio", type=float, default=None, help="Legacy strict-ablation knob: extra synchronization traffic ratio per extra replica.")
    parser.add_argument("--split_consistency_delay_s", type=float, default=0.0, help="Legacy strict-ablation knob: additional state-consistency barrier penalty (seconds).")
    parser.add_argument("--split_hop_traffic_overhead_ratio", type=float, default=0.0, help="[DEPRECATED legacy ablation] Use --split_sync_traffic_ratio instead.")
    parser.add_argument("--backhaul_knn_k", type=int, default=4, help="Backhaul topology sparsification by geography (0 disables sparsification / full mesh).")
    parser.add_argument("--backhaul_no_device_bw", action="store_true", help="Disable using device Bandwidth (MB/s); fall back to random BW generation for backhaul.")
    parser.add_argument("--backhaul_random_bw", action="store_true", help="Override and use random backhaul bandwidth instead of device bandwidth.")
    parser.add_argument("--backhaul_bw_scale", type=float, default=1.0, help="Scaling factor applied to device bandwidth when deriving backhaul BW.")
    parser.add_argument("--fiber_speed_mps", type=float, default=2e8, help="Propagation speed in fiber for backhaul (m/s).")
    parser.add_argument("--switch_delay_s", type=float, default=2e-4, help="Per-hop switching/queuing constant added to backhaul propagation latency (seconds).")
    parser.add_argument("--record_placement_trace", action="store_true", help="Write accepted request placements for post-placement robustness checks.")
    parser.add_argument("--pvfp_model_path", type=str, default=None, help="Path to a PVFP-FDRL checkpoint (.npz). If omitted, the simulator uses the configured fallback policy.")
    parser.add_argument("--pvfp_policy_mode", type=str, default="eval", choices=["eval", "train"], help="PVFP-FDRL policy mode recorded in metrics.")
    parser.add_argument("--pvfp_disable_fallback", action="store_true", help="Disable deterministic PVFP-inspired fallback when no checkpoint is found.")
    parser.add_argument("--out_dir", type=str, default="./result_online", help="Output directory for logs")
    args = parser.parse_args()

    spec = get_algorithm_spec(args.algo)
    if args.max_backup_level is None:
        args.max_backup_level = spec.default_max_backup_level
    if not (0 <= int(args.max_backup_level) <= 5):
        raise ValueError("--max_backup_level must be between 0 and 5")

    if args.mechanism is not None:
        if args.mechanism == "primary_only":
            args.diversity = False
            args.p_split = 0.0
            args.max_backup_level = 0
        elif args.mechanism == "redundancy_only":
            args.diversity = False
            args.p_split = 0.0
            args.max_backup_level = max(2, int(args.max_backup_level))
        elif args.mechanism == "split_only":
            args.diversity = True
            args.p_split = 1.0
            args.max_backup_level = 0
        elif args.mechanism == "dual":
            args.diversity = True
            args.max_backup_level = max(2, int(args.max_backup_level))
    if not (0.0 <= float(args.split_recovery_coverage) <= 1.0):
        raise ValueError("--split_recovery_coverage must be between 0 and 1")

    metrics, slot_log = simulate_online(
        edge_csv_path=args.edge_csv,
        region=args.region,
        time_slots=args.time_slots,
        arrival_rate=args.arrival_rate,
        mean_service_time=args.mean_service_time,
        k=args.k,
        seed=args.seed,
        enable_departure=(not args.no_departure),
        diversity=args.diversity,
        p_split=args.p_split,
        algo=args.algo,
        mechanism=args.mechanism,
        max_backup_level=args.max_backup_level,
        fixed_sfc_len=args.fixed_sfc_len,
        fixed_reliability=args.fixed_reliability,
        cap_base_min=args.cap_base_min,
        cap_base_max=args.cap_base_max,
        cap_scale=args.cap_scale,
        split_replica_count=args.split_replica_count,
        split_delay_overhead_ratio=args.split_delay_overhead_ratio,
        split_coord_delay_s=args.split_coord_delay_s,
        split_merge_delay_s=args.split_merge_delay_s,
        split_sync_traffic_ratio=(args.split_sync_traffic_ratio if args.split_sync_traffic_ratio is not None else args.split_hop_traffic_overhead_ratio),
        split_consistency_delay_s=args.split_consistency_delay_s,
        split_reliability_mode=args.split_reliability_mode,
        split_recovery_coverage=args.split_recovery_coverage,
        split_workload_shares_spec=args.split_workload_shares,
        backhaul_knn_k=args.backhaul_knn_k,
        backhaul_use_device_bw=(not args.backhaul_random_bw) and (not args.backhaul_no_device_bw),
        backhaul_bw_scale=args.backhaul_bw_scale,
        fiber_speed_mps=args.fiber_speed_mps,
        switch_delay_s=args.switch_delay_s,
        explicit_split_replicas=args.explicit_split_replicas,
        split_enforce_distinct=args.split_enforce_distinct,
        strict_split_resource_ablation=args.strict_split_resource_ablation,
        split_replica_resource_scale=args.split_replica_resource_scale,
        record_placement_trace=args.record_placement_trace,
        pvfp_model_path=args.pvfp_model_path,
        pvfp_policy_mode=args.pvfp_policy_mode,
        pvfp_fallback_if_missing=(not args.pvfp_disable_fallback),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    placement_trace = metrics.pop("_placement_trace", None)
    metrics_path = os.path.join(args.out_dir, f"metrics_{args.region}_{args.algo}_seed{args.seed}.json")
    if placement_trace is not None:
        placement_path = os.path.join(args.out_dir, f"placement_trace_{args.region}_{args.algo}_seed{args.seed}.json")
        with open(placement_path, "w", encoding="utf-8") as f:
            json.dump(placement_trace, f, ensure_ascii=False, indent=2)
        metrics["placement_trace_path"] = placement_path
    pd.Series(metrics).to_json(metrics_path, force_ascii=False, indent=2)

    slot_path = os.path.join(args.out_dir, f"slotlog_{args.region}_{args.algo}_seed{args.seed}.csv")
    pd.DataFrame(slot_log).to_csv(slot_path, index=False)

    print("=== Online simulation finished ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {slot_path}")


if __name__ == "__main__":
    main()
