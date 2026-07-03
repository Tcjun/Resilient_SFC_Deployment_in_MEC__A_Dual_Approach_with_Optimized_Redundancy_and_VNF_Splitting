#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from online_mdca_sim_delay_split_strict import simulate_online
from pvfp_fdrl_adapter import PVFP_SOURCE_PATH, PvfpFdrlConfig, PvfpFdrlPlacementPolicy, RewardWeights


def parse_int_csv(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def git_info() -> Dict[str, Any]:
    def _run(cmd: Sequence[str]) -> str:
        try:
            completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        except OSError:
            return ""
        return completed.stdout.strip()

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "status_short": _run(["git", "status", "--short"]),
    }


def reward_from_metrics(metrics: Dict[str, Any], weights: RewardWeights) -> float:
    accepted = float(metrics.get("total_accepted", 0.0))
    rejected = float(metrics.get("total_rejected", 0.0))
    avg_cost = float(metrics.get("avg_cost_per_accepted", 0.0))
    avg_delay = float(metrics.get("avg_e2e_delay_accepted", 0.0))
    qos = float(metrics.get("mdca_qos_rejected_total", metrics.get("qos_rejected_total", 0.0)))
    reliability = float(metrics.get("mdca_reliability_rejected_total", 0.0))
    capacity = float(metrics.get("mdca_capacity_rejected_total", 0.0))
    return (
        float(weights.accepted_reward) * accepted
        - float(weights.rejected_penalty) * rejected
        - float(weights.cost_penalty) * avg_cost
        - float(weights.delay_violation_penalty) * qos
        - float(weights.reliability_violation_penalty) * reliability
        - float(weights.capacity_violation_penalty) * capacity
        - float(weights.avg_delay_penalty) * avg_delay
    )


def simulate_kwargs(args: argparse.Namespace, *, region: str, seed: int, policy: PvfpFdrlPlacementPolicy) -> Dict[str, Any]:
    return {
        "edge_csv_path": args.edge_csv,
        "region": region,
        "time_slots": int(args.time_slots),
        "arrival_rate": float(args.arrival_rate),
        "mean_service_time": float(args.mean_service_time),
        "k": int(args.k),
        "seed": int(seed),
        "enable_departure": not bool(args.no_departure),
        "diversity": bool(args.diversity),
        "p_split": float(args.p_split),
        "algo": "pvfp_fdrl",
        "mechanism": args.mechanism,
        "max_backup_level": int(args.max_backup_level),
        "fixed_sfc_len": args.fixed_sfc_len,
        "fixed_reliability": args.fixed_reliability,
        "cap_base_min": int(args.cap_base_min),
        "cap_base_max": int(args.cap_base_max),
        "cap_scale": float(args.cap_scale),
        "split_replica_count": int(args.split_replica_count),
        "split_delay_overhead_ratio": float(args.split_delay_overhead_ratio),
        "split_coord_delay_s": float(args.split_coord_delay_s),
        "split_merge_delay_s": float(args.split_merge_delay_s),
        "split_sync_traffic_ratio": float(args.split_sync_traffic_ratio),
        "split_consistency_delay_s": float(args.split_consistency_delay_s),
        "split_reliability_mode": args.split_reliability_mode,
        "backhaul_knn_k": int(args.backhaul_knn_k),
        "backhaul_use_device_bw": not bool(args.backhaul_no_device_bw),
        "backhaul_bw_scale": float(args.backhaul_bw_scale),
        "fiber_speed_mps": float(args.fiber_speed_mps),
        "switch_delay_s": float(args.switch_delay_s),
        "explicit_split_replicas": bool(args.explicit_split_replicas),
        "split_enforce_distinct": bool(args.split_enforce_distinct),
        "strict_split_resource_ablation": bool(args.strict_split_resource_ablation),
        "split_replica_resource_scale": float(args.split_replica_resource_scale),
        "pvfp_policy_mode": str(policy.config.mode),
        "pvfp_fallback_if_missing": False,
        "pvfp_policy": policy,
    }


def evaluate_policy(
    policy: PvfpFdrlPlacementPolicy,
    *,
    regions: Sequence[str],
    seeds: Sequence[int],
    args: argparse.Namespace,
    reward_weights: RewardWeights,
) -> Tuple[float, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    rewards: List[float] = []
    for region in regions:
        for seed in seeds:
            metrics, _slot_log = simulate_online(**simulate_kwargs(args, region=region, seed=int(seed), policy=policy))
            reward = reward_from_metrics(metrics, reward_weights)
            rewards.append(float(reward))
            rows.append(
                {
                    "region": region,
                    "seed": int(seed),
                    "reward": float(reward),
                    "acceptance_ratio": float(metrics.get("acceptance_ratio", 0.0)),
                    "avg_cost_per_accepted": float(metrics.get("avg_cost_per_accepted", 0.0)),
                    "p95_e2e_delay_accepted": float(metrics.get("p95_e2e_delay_accepted", 0.0)),
                    "capacity_rejections": int(metrics.get("mdca_capacity_rejected_total", 0)),
                    "reliability_rejections": int(metrics.get("mdca_reliability_rejected_total", 0)),
                    "qos_rejections": int(metrics.get("mdca_qos_rejected_total", 0)),
                    "trace_hash": metrics.get("trace_hash", ""),
                }
            )
    return float(np.mean(rewards)) if rewards else 0.0, rows


def write_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def build_eval_command_template(args: argparse.Namespace, checkpoint_path: Path) -> str:
    parts = [
        sys.executable,
        "online_mdca_sim_delay_split_strict.py",
        "--edge_csv",
        str(Path(args.edge_csv).resolve()),
        "--region",
        "<REGION>",
        "--time_slots",
        str(args.time_slots),
        "--arrival_rate",
        str(args.arrival_rate),
        "--mean_service_time",
        str(args.mean_service_time),
        "--k",
        str(args.k),
        "--seed",
        "<TEST_SEED>",
        "--algo",
        "pvfp_fdrl",
        "--max_backup_level",
        str(args.max_backup_level),
        "--cap_scale",
        str(args.cap_scale),
        "--pvfp_model_path",
        str(checkpoint_path.resolve()),
    ]
    if args.diversity:
        parts += [
            "--diversity",
            "--p_split",
            str(args.p_split),
            "--explicit_split_replicas",
            "--split_replica_count",
            str(args.split_replica_count),
            "--split_reliability_mode",
            args.split_reliability_mode,
        ]
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_csv", type=str, default="Edge_devices.csv")
    parser.add_argument("--region", type=str, default="Alabama")
    parser.add_argument("--regions", type=str, default="", help="Comma-separated training regions. If omitted, --region is used.")
    parser.add_argument("--train_seeds", type=str, default="101,102")
    parser.add_argument("--validation_seeds", type=str, default="201,202")
    parser.add_argument("--test_seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--time_slots", type=int, default=30)
    parser.add_argument("--arrival_rate", type=float, default=10.0)
    parser.add_argument("--mean_service_time", type=float, default=5.0)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--max_backup_level", type=int, default=2)
    parser.add_argument("--fixed_sfc_len", type=int, default=None)
    parser.add_argument("--fixed_reliability", type=float, default=None)
    parser.add_argument("--mechanism", type=str, default=None, choices=["primary_only", "redundancy_only", "split_only", "dual"])
    parser.add_argument("--no_departure", action="store_true")
    parser.add_argument("--diversity", action="store_true")
    parser.add_argument("--p_split", type=float, default=0.3)
    parser.add_argument("--explicit_split_replicas", action="store_true")
    parser.add_argument("--split_enforce_distinct", action="store_true")
    parser.add_argument("--strict_split_resource_ablation", action="store_true")
    parser.add_argument("--split_replica_resource_scale", type=float, default=1.0)
    parser.add_argument("--split_replica_count", type=int, default=2)
    parser.add_argument("--split_reliability_mode", type=str, default="identical_active_pool", choices=["identical_active_pool", "heterogeneous_upper_bound"])
    parser.add_argument("--split_delay_overhead_ratio", type=float, default=0.15)
    parser.add_argument("--split_coord_delay_s", type=float, default=0.0)
    parser.add_argument("--split_merge_delay_s", type=float, default=0.0)
    parser.add_argument("--split_sync_traffic_ratio", type=float, default=0.0)
    parser.add_argument("--split_consistency_delay_s", type=float, default=0.0)
    parser.add_argument("--backhaul_knn_k", type=int, default=4)
    parser.add_argument("--backhaul_no_device_bw", action="store_true")
    parser.add_argument("--backhaul_bw_scale", type=float, default=1.0)
    parser.add_argument("--fiber_speed_mps", type=float, default=2e8)
    parser.add_argument("--switch_delay_s", type=float, default=2e-4)
    parser.add_argument("--cap_base_min", type=int, default=500)
    parser.add_argument("--cap_base_max", type=int, default=700)
    parser.add_argument("--cap_scale", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--action_dim", type=int, default=512)
    parser.add_argument("--num_domains", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--aggregation_interval", type=int, default=32)
    parser.add_argument("--local_updates_per_decision", type=int, default=1)
    parser.add_argument("--base_aggregation_weight", type=float, default=0.2)
    parser.add_argument("--lambda_staleness", type=float, default=5.0)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--checkpoint_out", type=str, default="result_pvfp_fdrl/pvfp_fdrl_checkpoint.npz")
    parser.add_argument("--metadata_out", type=str, default="result_pvfp_fdrl/pvfp_fdrl_training_metadata.json")
    parser.add_argument("--accepted_reward", type=float, default=1.0)
    parser.add_argument("--rejected_penalty", type=float, default=1.0)
    parser.add_argument("--cost_penalty", type=float, default=0.001)
    parser.add_argument("--delay_violation_penalty", type=float, default=2.0)
    parser.add_argument("--reliability_violation_penalty", type=float, default=3.0)
    parser.add_argument("--capacity_violation_penalty", type=float, default=2.0)
    parser.add_argument("--avg_delay_penalty", type=float, default=0.0)
    args = parser.parse_args()

    train_seeds = parse_int_csv(args.train_seeds)
    validation_seeds = parse_int_csv(args.validation_seeds)
    test_seeds = parse_int_csv(args.test_seeds)
    regions = parse_str_csv(args.regions) if args.regions else [args.region]
    overlap = set(train_seeds) & (set(validation_seeds) | set(test_seeds))
    if overlap:
        raise ValueError(f"Train seeds must be disjoint from validation/test seeds: {sorted(overlap)}")

    reward_weights = RewardWeights(
        accepted_reward=args.accepted_reward,
        rejected_penalty=args.rejected_penalty,
        cost_penalty=args.cost_penalty,
        delay_violation_penalty=args.delay_violation_penalty,
        reliability_violation_penalty=args.reliability_violation_penalty,
        capacity_violation_penalty=args.capacity_violation_penalty,
        avg_delay_penalty=args.avg_delay_penalty,
    )
    checkpoint_out = Path(args.checkpoint_out).resolve()
    metadata_out = Path(args.metadata_out).resolve()

    config = PvfpFdrlConfig(
        checkpoint_path=(args.resume_checkpoint or None),
        mode="train",
        seed=int(train_seeds[0] if train_seeds else 1),
        num_domains=int(args.num_domains),
        action_dim=int(args.action_dim),
        batch_size=int(args.batch_size),
        aggregation_interval=int(args.aggregation_interval),
        local_updates_per_decision=int(args.local_updates_per_decision),
        base_aggregation_weight=float(args.base_aggregation_weight),
        lambda_staleness=float(args.lambda_staleness),
        fallback_if_missing=False,
        reward_weights=reward_weights,
    )
    policy = PvfpFdrlPlacementPolicy(config)
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(args.epochs) + 1):
        policy.config.mode = "train"
        train_reward, train_rows = evaluate_policy(policy, regions=regions, seeds=train_seeds, args=args, reward_weights=reward_weights)
        policy.aggregate()
        policy.config.mode = "eval"
        validation_reward, validation_rows = evaluate_policy(
            policy,
            regions=regions,
            seeds=validation_seeds,
            args=args,
            reward_weights=reward_weights,
        )
        history.append(
            {
                "epoch": int(epoch),
                "mean_train_reward": float(train_reward),
                "train_rows": train_rows,
                "validation_mean_reward": float(validation_reward),
                "validation_rows": validation_rows,
                "policy_status": policy.status(),
            }
        )
        print(
            f"epoch={epoch} mean_train_reward={train_reward:.6f} "
            f"validation_mean_reward={validation_reward:.6f} "
            f"training_steps={policy.status().get('training_steps', 0)} "
            f"aggregation_rounds={policy.status().get('aggregation_rounds', 0)}",
            flush=True,
        )

    policy.config.mode = "eval"
    validation_reward, validation_rows = evaluate_policy(
        policy,
        regions=regions,
        seeds=validation_seeds,
        args=args,
        reward_weights=reward_weights,
    )

    metadata: Dict[str, Any] = {
        "algorithm": "PVFP-FDRL",
        "implementation_boundary": "PVFP DQN/FDRL code adapted to the placement stage of the existing online simulator",
        "training_algorithm": "PVFP-backed domain DQN with replay buffer and federated aggregation",
        "pvfp_source_path": str(PVFP_SOURCE_PATH),
        "created_at_unix": time.time(),
        "git": git_info(),
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "test_seeds": test_seeds,
        "regions": regions,
        "arrival_rate": args.arrival_rate,
        "time_slots": args.time_slots,
        "mean_service_time": args.mean_service_time,
        "sfc_length": args.fixed_sfc_len,
        "reliability_target": args.fixed_reliability,
        "p_split": args.p_split if args.diversity else None,
        "max_backup_level": args.max_backup_level,
        "split_replica_count": args.split_replica_count,
        "reward_weights": asdict(reward_weights),
        "model_hyperparameters": {
            "state_dim": config.state_dim,
            "action_dim": config.action_dim,
            "num_domains": config.num_domains,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "aggregation_interval": args.aggregation_interval,
            "local_updates_per_decision": args.local_updates_per_decision,
            "base_aggregation_weight": args.base_aggregation_weight,
            "lambda_staleness": args.lambda_staleness,
        },
        "checkpoint_path": str(checkpoint_out),
        "metadata_path": str(metadata_out),
        "training_command": " ".join(sys.argv),
        "evaluation_command_template": build_eval_command_template(args, checkpoint_out),
        "fallback_policy_used_during_training": False,
        "history": history,
        "validation_mean_reward": validation_reward,
        "validation_rows": validation_rows,
        "final_policy_status": policy.status(),
    }

    policy.save_checkpoint(checkpoint_out, metadata=metadata)
    write_metadata(metadata_out, metadata)
    print(f"Saved checkpoint: {checkpoint_out}")
    print(f"Saved metadata: {metadata_out}")
    print(f"validation_mean_reward: {validation_reward:.6f}")


if __name__ == "__main__":
    main()
