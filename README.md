# Resilient SFC Deployment in MEC

This repository contains the public experimental artifact for the paper:

**Resilient SFC Deployment in MEC: A Dual Approach with Optimized Redundancy and VNF Splitting**

The code evaluates resilient service function chain (SFC) deployment in multi-access edge computing (MEC) environments. It includes the online simulator, the MDCA/DMDCA experiment runners, benchmark adapters, and precomputed result files used for the paper's experimental analysis.

## Scope of This Release

This repository is a clean public release of the experiment code and result data. It intentionally does not include:

- Git history from the private development workspace
- LaTeX manuscript source files
- Review letters or revision notes
- Third-party reference papers
- Plotting scripts or manuscript figure-generation scripts
- Local build artifacts, temporary files, or IDE/Codex metadata

The precomputed CSV/JSON outputs under `paper_data/` are included so that the reported numerical results can be inspected without rerunning all experiments.

## Repository Layout

```text
.
|-- online_mdca_sim_delay_split_strict.py   # Core online MEC/SFC simulator
|-- run_online_paper_experiments.py         # Main online experiment runner
|-- run_sensitivity_sweep.py                # General sensitivity runner
|-- run_recovery_coverage_sensitivity.py    # Recovery-coverage eta sweep
|-- run_workload_sharing_sensitivity.py     # Workload-share sensitivity
|-- run_scalability_runtime.py              # Runtime scalability experiments
|-- run_small_scale_opt_reference.py        # Compact exact-reference checks
|-- run_large_scale_lb_reference.py         # Trace-level lower-bound reference
|-- run_correlated_failure_stress.py        # Post-placement correlated-failure stress check
|-- train_pvfp_fdrl_baseline.py             # PVFP-FDRL training/evaluation wrapper
|-- pvfp_fdrl_adapter.py                    # PVFP-FDRL adapter used by the simulator
|-- Alabama_data.csv                        # Region-level input data
|-- Arizona_data.csv                        # Region-level input data
|-- Edge_devices.csv                        # Edge-device/cloudlet input data
`-- paper_data/                             # Precomputed experimental results
```

## Precomputed Results

The main result folders are:

```text
paper_data/full_online_runs/
paper_data/m_sensitivity/
paper_data/recovery_coverage_sensitivity/
paper_data/workload_sharing_sensitivity/
paper_data/split_count_sensitivity/
paper_data/scalability_runtime_lambda30/
paper_data/small_scale_opt_reference/
paper_data/small_scale_opt_reference_300s/
paper_data/large_scale_lb_reference/
paper_data/correlated_failure_stress/
paper_data/mobility_stress/
paper_data/ci_reporting_check/
```

Useful summary files include:

```text
paper_data/full_online_runs/summary_metrics_with_pvfp.csv
paper_data/full_online_runs/raw_metrics_long.csv
paper_data/m_sensitivity/summary_metrics.csv
paper_data/recovery_coverage_sensitivity/summary_metrics.csv
paper_data/workload_sharing_sensitivity/summary_metrics.csv
paper_data/split_count_sensitivity/summary_metrics.csv
paper_data/scalability_runtime_lambda30/summary_metrics.csv
paper_data/small_scale_opt_reference/summary.csv
paper_data/large_scale_lb_reference/summary.csv
paper_data/correlated_failure_stress/summary_stress_metrics.csv
paper_data/mobility_stress/summary_mobility_metrics.csv
```

Many result directories also include `commands.txt`, `run_config.json`, `fairness_checks.csv`, raw per-seed metrics, and placement traces. These files are provided to make the reported aggregates auditable.

## Environment

The experiments were run with Python 3 and use standard scientific Python packages.

Minimal dependencies for the released scripts:

```bash
pip install numpy pandas
```

Some optional analysis or plotting code in a local environment may require additional packages, but plotting scripts are not part of this public release.

## Quick Checks

From the repository root:

```bash
python online_mdca_sim_delay_split_strict.py --help
python run_online_paper_experiments.py --help
python run_scalability_runtime.py --help
```

To inspect the main precomputed online results:

```python
import pandas as pd

df = pd.read_csv("paper_data/full_online_runs/summary_metrics_with_pvfp.csv")
print(df.head())
print(df.columns.tolist())
```

## Reproducing Experiments

The main online experiments can be launched with:

```bash
python run_online_paper_experiments.py \
  --use_current_python \
  --out_root paper_data/full_online_runs_reproduced
```

The default runner covers the main regions, seeds, load sweep, SFC-length sweep, and reliability sweep used by the paper. Because the full experiment set can be time-consuming, first use a smaller smoke run:

```bash
python run_online_paper_experiments.py \
  --use_current_python \
  --max_points_per_experiment 1 \
  --seeds 1 \
  --out_root paper_data/smoke_check
```

Additional experiment runners:

```bash
python run_recovery_coverage_sensitivity.py --help
python run_workload_sharing_sensitivity.py --help
python run_scalability_runtime.py --help
python run_small_scale_opt_reference.py --help
python run_large_scale_lb_reference.py --help
python run_correlated_failure_stress.py --help
```

## Notes on PVFP-FDRL

The precomputed main summary file includes PVFP-FDRL comparison rows. The adapter in `pvfp_fdrl_adapter.py` expects an external source file at:

```text
PVFP/pvfp_fed_dqn.py
```

If that file is not present in your checkout, the included precomputed PVFP-FDRL metrics can still be inspected, but rerunning or retraining the PVFP-FDRL baseline requires adding the corresponding baseline source file.

## Interpreting the Results

The released experiments evaluate:

- MDCA under standby-only resilient SFC placement
- DMDCA under fixed two-replica active load-sharing VNF splitting
- Sensitivity to split activation probability, active replica count, workload share, backup budget, and recovery coverage
- Runtime scalability with network size, request load, and SFC length
- Compact exact-reference checks and trace-level lower-bound checks
- Post-placement stress checks for correlated failures and mobility-induced ingress changes

The correlated-failure and mobility results are post-placement robustness checks. They should not be interpreted as correlated-failure-aware or mobility-aware optimization.

## Citation

If you use this artifact, please cite the associated paper once bibliographic information is available.

## License

This repository is released under the Apache License 2.0. See `LICENSE` for details.
