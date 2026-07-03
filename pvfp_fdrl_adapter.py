#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


PVFP_SOURCE_PATH = Path(__file__).resolve().parent / "PVFP" / "pvfp_fed_dqn.py"

STATE_FEATURE_NAMES = [
    "candidate_count_ratio",
    "capacity_mean_ratio",
    "capacity_min_ratio",
    "residual_after_mean_ratio",
    "residual_after_min_ratio",
    "cost_min_ratio",
    "cost_mean_ratio",
    "cloudlet_cost_mean_ratio",
    "access_delay_ratio",
    "prev_link_delay_min_ratio",
    "prev_link_delay_mean_ratio",
    "prev_link_bandwidth_max_ratio",
    "resource_need_ratio",
    "is_backup",
    "is_split_replica",
    "chain_position_ratio",
    "placed_instance_ratio",
    "required_reliability",
    "vnf_reliability",
    "traffic_ratio",
]


@dataclass
class RewardWeights:
    accepted_reward: float = 1.0
    rejected_penalty: float = 1.0
    cost_penalty: float = 0.001
    delay_violation_penalty: float = 2.0
    reliability_violation_penalty: float = 3.0
    capacity_violation_penalty: float = 2.0
    avg_delay_penalty: float = 0.0


@dataclass
class PvfpFdrlConfig:
    checkpoint_path: Optional[str] = None
    mode: str = "eval"
    seed: int = 1
    num_domains: int = 3
    state_dim: int = len(STATE_FEATURE_NAMES)
    action_dim: int = 512
    fallback_if_missing: bool = True
    aggregation_interval: int = 32
    local_updates_per_decision: int = 1
    batch_size: int = 64
    base_aggregation_weight: float = 0.2
    lambda_staleness: float = 5.0
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    capacity_weight: float = 1.0
    cost_weight: float = 0.7
    delay_weight: float = 0.8
    bandwidth_weight: float = 0.15


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _normalizer(values: np.ndarray, floor: float = 1e-9) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    return float(max(np.max(np.abs(finite)), floor))


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def load_pvfp_module():
    if not torch_available():
        raise RuntimeError(
            "PyTorch is required for the PVFP-backed FDRL policy. "
            "Install torch on the training/evaluation environment, or allow fallback for interface-only smoke tests."
        )
    if not PVFP_SOURCE_PATH.exists():
        raise FileNotFoundError(f"PVFP source file not found: {PVFP_SOURCE_PATH}")
    spec = importlib.util.spec_from_file_location("pvfp_fed_dqn_source", PVFP_SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PVFP module from {PVFP_SOURCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def partition_cloudlets_into_domains(num_cloudlets: int, num_domains: int = 3) -> Dict[int, List[int]]:
    domains: Dict[int, List[int]] = {domain_id: [] for domain_id in range(max(1, int(num_domains)))}
    for c_idx in range(int(num_cloudlets)):
        domains[c_idx % len(domains)].append(int(c_idx))
    return domains


def _previous_primary_nodes(
    chain: Sequence[int],
    v_id: int,
    placed_instances: Sequence[Tuple[int, int, int, float, int]],
) -> Tuple[List[int], int]:
    chain_int = list(map(int, chain))
    try:
        pos = chain_int.index(int(v_id))
    except ValueError:
        pos = -1
    if pos <= 0:
        return [], max(pos, 0)
    prev_vnf = int(chain_int[pos - 1])
    nodes: List[int] = []
    for inst in placed_instances:
        inst_v, is_backup, c_idx, _res_need, _rep_idx = inst
        if int(inst_v) == prev_vnf and int(is_backup) == 0:
            nodes.append(int(c_idx))
    return nodes, pos


def build_state_from_simulator(
    *,
    candidate_ids: Sequence[int],
    cap_left: np.ndarray,
    res_need: float,
    base_need: float,
    cost_row: np.ndarray,
    vct: np.ndarray,
    access_delay_s: float,
    latency_budget_s: float,
    L: np.ndarray,
    BW: np.ndarray,
    traffic_bits: float,
    req_reliability: float,
    vnf_reliability: float,
    v_id: int,
    is_backup: int,
    rep_idx: int,
    chain: Sequence[int],
    placed_instances: Sequence[Tuple[int, int, int, float, int]],
) -> np.ndarray:
    candidates = np.asarray(candidate_ids, dtype=int)
    if candidates.size == 0:
        return np.zeros(len(STATE_FEATURE_NAMES), dtype=np.float32)

    cap = np.asarray(cap_left, dtype=float)
    costs = np.asarray(cost_row, dtype=float)
    vct_arr = np.asarray(vct, dtype=float)
    latency = np.asarray(L, dtype=float)
    bandwidth = np.asarray(BW, dtype=float)

    cap_norm = _normalizer(cap)
    candidate_cap = cap[candidates]
    residual_after = np.maximum(candidate_cap - float(res_need), 0.0)
    candidate_cost = costs[candidates] if costs.size else np.zeros_like(candidates, dtype=float)
    candidate_vct = vct_arr[candidates] if vct_arr.size else np.zeros_like(candidates, dtype=float)

    cost_norm = _normalizer(candidate_cost)
    vct_norm = _normalizer(vct_arr)
    latency_norm = _normalizer(latency)
    bandwidth_norm = _normalizer(bandwidth)
    resource_norm = max(_safe_float(base_need, 0.0), _safe_float(res_need, 0.0), 1e-9)
    budget_norm = max(_safe_float(latency_budget_s, 0.0), 1e-9)
    traffic_norm = max(_safe_float(traffic_bits, 0.0), 1e-9)

    prev_nodes, chain_pos = _previous_primary_nodes(chain, int(v_id), placed_instances)
    prev_delay_values: List[float] = []
    prev_bw_values: List[float] = []
    for c_idx in candidates:
        c = int(c_idx)
        for p in prev_nodes:
            if 0 <= p < latency.shape[0] and 0 <= c < latency.shape[1]:
                prev_delay_values.append(float(latency[p, c]))
            if 0 <= p < bandwidth.shape[0] and 0 <= c < bandwidth.shape[1]:
                prev_bw_values.append(float(bandwidth[p, c]))
    prev_delay_arr = np.asarray(prev_delay_values, dtype=float) if prev_delay_values else np.asarray([0.0], dtype=float)
    prev_bw_arr = np.asarray(prev_bw_values, dtype=float) if prev_bw_values else np.asarray([0.0], dtype=float)

    chain_len = max(len(chain), 1)
    state = np.asarray(
        [
            float(len(candidates) / max(len(cap), 1)),
            float(np.mean(candidate_cap) / cap_norm),
            float(np.min(candidate_cap) / cap_norm),
            float(np.mean(residual_after) / cap_norm),
            float(np.min(residual_after) / cap_norm),
            float(np.min(candidate_cost) / cost_norm) if candidate_cost.size else 0.0,
            float(np.mean(candidate_cost) / cost_norm) if candidate_cost.size else 0.0,
            float(np.mean(candidate_vct) / vct_norm) if candidate_vct.size else 0.0,
            float(_safe_float(access_delay_s) / budget_norm),
            float(np.min(prev_delay_arr) / latency_norm),
            float(np.mean(prev_delay_arr) / latency_norm),
            float(np.max(prev_bw_arr) / bandwidth_norm),
            float(_safe_float(res_need) / resource_norm),
            1.0 if int(is_backup) != 0 else 0.0,
            1.0 if int(rep_idx) >= 0 else 0.0,
            float(max(chain_pos, 0) / chain_len),
            float(len(placed_instances) / max(chain_len, 1)),
            float(np.clip(_safe_float(req_reliability), 0.0, 1.0)),
            float(np.clip(_safe_float(vnf_reliability), 0.0, 1.0)),
            float(_safe_float(traffic_bits) / traffic_norm),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)


def request_reward(outcome: Dict[str, Any], weights: RewardWeights) -> float:
    reason = str(outcome.get("reason", "accepted"))
    accepted = bool(outcome.get("accepted", False))
    reward = float(weights.accepted_reward) if accepted else -float(weights.rejected_penalty)
    reward -= float(weights.cost_penalty) * _safe_float(outcome.get("cost", 0.0), 0.0)
    reward -= float(weights.avg_delay_penalty) * _safe_float(outcome.get("e2e_delay", 0.0), 0.0)
    if reason == "qos":
        reward -= float(weights.delay_violation_penalty)
    elif reason == "reliability":
        reward -= float(weights.reliability_violation_penalty)
    elif reason == "capacity":
        reward -= float(weights.capacity_violation_penalty)
    return float(reward)


class PvfpFdrlPlacementPolicy:
    def __init__(self, config: Optional[PvfpFdrlConfig] = None) -> None:
        self.config = config or PvfpFdrlConfig()
        self.rng = np.random.default_rng(int(self.config.seed))
        self.loaded_checkpoint = False
        self.fallback_active = False
        self.fallback_used = False
        self.initialized = False
        self.pvfp_module = None
        self.agents: List[Any] = []
        self.server: Any = None
        self.action_dim = int(self.config.action_dim)
        self.domain_map: Dict[int, List[int]] = {}
        self.current_req_id: Optional[int] = None
        self.pending: Dict[int, List[Dict[str, Any]]] = {}
        self.training_steps = 0
        self.aggregation_rounds = 0
        self.loss_history: List[float] = []

    def _ensure_backend(self, min_action_dim: int) -> bool:
        required_action_dim = max(int(self.config.action_dim), int(min_action_dim))
        if self.initialized and required_action_dim <= self.action_dim:
            return True
        if not torch_available():
            if self.config.fallback_if_missing:
                self.fallback_active = True
                return False
            load_pvfp_module()

        self.pvfp_module = load_pvfp_module()
        self.action_dim = required_action_dim
        self.domain_map = partition_cloudlets_into_domains(self.action_dim, self.config.num_domains)

        self.agents = []
        for domain_id in range(int(self.config.num_domains)):
            env = self.pvfp_module.DomainEnv(domain_id, num_nodes=4)
            env.replay_capacity = max(64, int(self.config.batch_size) * 4)
            agent = self.pvfp_module.DQNAgent(
                env,
                input_dim=int(self.config.state_dim),
                action_dim=int(self.action_dim),
                batch_size=int(self.config.batch_size),
            )
            self.agents.append(agent)

        self.server = self.pvfp_module.FederatedServer(
            base_model=self.agents[0].q_net,
            lambda_staleness=float(self.config.lambda_staleness),
            base_delta=float(self.config.base_aggregation_weight),
        )
        self.initialized = True

        checkpoint_path = self.config.checkpoint_path
        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)
        return True

    def _domain_for_decision(self, v_id: int, rep_idx: int) -> int:
        offset = max(int(rep_idx), 0)
        return int((int(v_id) + offset) % max(int(self.config.num_domains), 1))

    def begin_request(self, req_id: int) -> None:
        self.current_req_id = int(req_id)
        self.pending.setdefault(int(req_id), [])

    def _fallback_scores(
        self,
        *,
        feasible: Sequence[int],
        cap_left: np.ndarray,
        res_need: float,
        cost_row: np.ndarray,
        vct: np.ndarray,
        L: np.ndarray,
        BW: np.ndarray,
        placed_instances: Sequence[Tuple[int, int, int, float, int]],
    ) -> np.ndarray:
        self.fallback_used = True
        candidates = np.asarray(feasible, dtype=int)
        cap = np.asarray(cap_left, dtype=float)
        costs = np.asarray(cost_row, dtype=float)
        vct_arr = np.asarray(vct, dtype=float)
        latency = np.asarray(L, dtype=float)
        bandwidth = np.asarray(BW, dtype=float)
        residual = np.maximum(cap[candidates] - float(res_need), 0.0)
        cost = costs[candidates] if costs.size else np.zeros_like(candidates, dtype=float)
        cloud_cost = vct_arr[candidates] if vct_arr.size else np.zeros_like(candidates, dtype=float)
        if placed_instances:
            prev_nodes = [int(inst[2]) for inst in placed_instances if int(inst[1]) == 0]
            delay_proxy = np.asarray(
                [min(float(latency[p, c]) for p in prev_nodes) if prev_nodes else 0.0 for c in candidates],
                dtype=float,
            )
            bw_proxy = np.asarray(
                [max(float(bandwidth[p, c]) for p in prev_nodes) if prev_nodes else 0.0 for c in candidates],
                dtype=float,
            )
        else:
            delay_proxy = np.zeros_like(candidates, dtype=float)
            bw_proxy = np.zeros_like(candidates, dtype=float)
        return (
            float(self.config.capacity_weight) * residual / _normalizer(cap)
            + float(self.config.bandwidth_weight) * bw_proxy / _normalizer(bandwidth)
            - float(self.config.cost_weight) * (cost / _normalizer(cost))
            - float(self.config.delay_weight) * (delay_proxy / _normalizer(latency))
            - 0.2 * (cloud_cost / _normalizer(vct_arr))
        )

    def choose_cloudlet(
        self,
        *,
        candidate_ids: Sequence[int],
        cap_left: np.ndarray,
        res_need: float,
        base_need: float,
        cost_row: np.ndarray,
        vct: np.ndarray,
        access_delay_s: float,
        latency_budget_s: float,
        L: np.ndarray,
        BW: np.ndarray,
        traffic_bits: float,
        req_reliability: float,
        vnf_reliability: float,
        v_id: int,
        is_backup: int,
        rep_idx: int,
        chain: Sequence[int],
        placed_instances: Sequence[Tuple[int, int, int, float, int]],
    ) -> Optional[int]:
        feasible = [int(c_idx) for c_idx in candidate_ids if float(cap_left[int(c_idx)]) >= float(res_need)]
        if not feasible:
            return None

        min_action_dim = max(max(feasible) + 1, len(cap_left))
        backend_ready = self._ensure_backend(min_action_dim)
        state = build_state_from_simulator(
            candidate_ids=feasible,
            cap_left=cap_left,
            res_need=res_need,
            base_need=base_need,
            cost_row=cost_row,
            vct=vct,
            access_delay_s=access_delay_s,
            latency_budget_s=latency_budget_s,
            L=L,
            BW=BW,
            traffic_bits=traffic_bits,
            req_reliability=req_reliability,
            vnf_reliability=vnf_reliability,
            v_id=v_id,
            is_backup=is_backup,
            rep_idx=rep_idx,
            chain=chain,
            placed_instances=placed_instances,
        )

        if backend_ready:
            domain_id = self._domain_for_decision(int(v_id), int(rep_idx))
            agent = self.agents[domain_id]
            training = str(self.config.mode).lower() == "train"
            action = int(agent.select_action(state, valid_actions=feasible, training=training))
            if action not in feasible:
                action = int(feasible[0])
            if training and self.current_req_id is not None:
                self.pending.setdefault(int(self.current_req_id), []).append(
                    {
                        "domain_id": int(domain_id),
                        "state": state.copy(),
                        "action": int(action),
                    }
                )
            return int(action)

        scores = self._fallback_scores(
            feasible=feasible,
            cap_left=cap_left,
            res_need=res_need,
            cost_row=cost_row,
            vct=vct,
            L=L,
            BW=BW,
            placed_instances=placed_instances,
        )
        return int(feasible[int(np.nanargmax(scores))])

    def finish_request(self, req_id: int, outcome: Dict[str, Any]) -> None:
        if str(self.config.mode).lower() != "train" or not self.initialized:
            self.pending.pop(int(req_id), None)
            return
        decisions = self.pending.pop(int(req_id), [])
        if not decisions:
            return
        reward = request_reward(outcome, self.config.reward_weights)
        losses: List[float] = []
        for decision in decisions:
            domain_id = int(decision["domain_id"])
            agent = self.agents[domain_id]
            state = np.asarray(decision["state"], dtype=np.float32)
            action = int(decision["action"])
            agent.push_transition(state, action, float(reward), state.copy(), True)
            for _ in range(max(1, int(self.config.local_updates_per_decision))):
                loss = agent.update()
                if loss is not None:
                    losses.append(float(loss))
            agent.soft_update_target()
            agent.adapt_epsilon_by_reward(float(reward))
        self.training_steps += len(decisions)
        self.loss_history.extend(losses[-32:])
        if self.training_steps > 0 and self.training_steps % max(1, int(self.config.aggregation_interval)) == 0:
            self.aggregate()

    def aggregate(self) -> None:
        if not self.initialized or self.server is None:
            return
        uploads = []
        now = time.time()
        for idx, agent in enumerate(self.agents):
            uploads.append((now + idx * 0.001, copy.deepcopy(agent.q_net)))
        self.server.aggregate(uploads)
        global_model = self.server.distribute()
        for agent in self.agents:
            agent.q_net.load_state_dict(global_model.state_dict())
            agent.target_q.load_state_dict(global_model.state_dict())
        self.aggregation_rounds += 1

    def save_checkpoint(self, path: str | os.PathLike[str], *, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.initialized:
            self._ensure_backend(self.action_dim)
        if not self.initialized:
            raise RuntimeError("Cannot save a PVFP checkpoint without an initialized PVFP backend.")
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch = self.pvfp_module.torch
        payload = {
            "format": "pvfp_fdrl_torch_v1",
            "pvfp_source_path": str(PVFP_SOURCE_PATH),
            "config": asdict(self.config),
            "action_dim": int(self.action_dim),
            "state_dim": int(self.config.state_dim),
            "agent_q_state_dicts": [agent.q_net.state_dict() for agent in self.agents],
            "agent_target_state_dicts": [agent.target_q.state_dict() for agent in self.agents],
            "agent_epsilons": [float(agent.epsilon) for agent in self.agents],
            "server_state_dict": self.server.global_model.state_dict() if self.server is not None else None,
            "training_steps": int(self.training_steps),
            "aggregation_rounds": int(self.aggregation_rounds),
            "metadata": metadata or {},
        }
        torch.save(payload, checkpoint_path)

    def load_checkpoint(self, path: str | os.PathLike[str]) -> Dict[str, Any]:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(str(checkpoint_path))
        if self.pvfp_module is None:
            self.pvfp_module = load_pvfp_module()
        torch = self.pvfp_module.torch
        payload = torch.load(checkpoint_path, map_location=self.pvfp_module.DEVICE)
        if payload.get("format") != "pvfp_fdrl_torch_v1":
            raise ValueError(f"Unsupported PVFP checkpoint format: {payload.get('format')}")
        loaded_action_dim = int(payload.get("action_dim", self.action_dim))
        if loaded_action_dim > self.action_dim:
            self.action_dim = loaded_action_dim
        if not self.initialized:
            self._ensure_backend(self.action_dim)
        for agent, state in zip(self.agents, payload.get("agent_q_state_dicts", [])):
            agent.q_net.load_state_dict(state)
        for agent, state in zip(self.agents, payload.get("agent_target_state_dicts", [])):
            agent.target_q.load_state_dict(state)
        for agent, epsilon in zip(self.agents, payload.get("agent_epsilons", [])):
            agent.epsilon = float(epsilon)
        if self.server is not None and payload.get("server_state_dict") is not None:
            self.server.global_model.load_state_dict(payload["server_state_dict"])
        self.training_steps = int(payload.get("training_steps", self.training_steps))
        self.aggregation_rounds = int(payload.get("aggregation_rounds", self.aggregation_rounds))
        self.loaded_checkpoint = True
        return dict(payload.get("metadata", {}))

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "pvfp_torch" if self.initialized else ("fallback" if self.fallback_active else "uninitialized"),
            "uses_pvfp_code": bool(self.initialized),
            "pvfp_source_path": str(PVFP_SOURCE_PATH),
            "torch_available": bool(torch_available()),
            "loaded_checkpoint": bool(self.loaded_checkpoint),
            "fallback_active": bool(self.fallback_active),
            "fallback_used": bool(self.fallback_used),
            "checkpoint_path": self.config.checkpoint_path,
            "mode": self.config.mode,
            "training_steps": int(self.training_steps),
            "aggregation_rounds": int(self.aggregation_rounds),
            "state_features": list(STATE_FEATURE_NAMES),
        }
