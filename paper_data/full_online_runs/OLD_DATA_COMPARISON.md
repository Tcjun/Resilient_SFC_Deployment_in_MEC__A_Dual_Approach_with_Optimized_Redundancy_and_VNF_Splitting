# Old Data Comparison

This note compares the authoritative data in `paper_data/full_online_runs/summary_metrics.csv` with the previous plot-ready source:

```text
paper_online_runs_split_calibrated/summary_metrics_plot_ready.csv
```

Section names were mapped as follows for comparison:

```text
old mdca  -> new unsplit
old dmdca -> new split_enabled
```

## Summary

The unsplit results used in Figures 3--6 are effectively consistent with the previous plot-ready data. The largest absolute differences across the plotted means are small:

```text
unsplit/load:
  acceptance_ratio_mean max abs diff              0.00728377
  avg_cost_per_accepted_mean max abs diff         0.439887
  p95_e2e_delay_accepted_mean max abs diff        0.00243815
  qos_rejection_ratio_mean max abs diff           0.00121029

unsplit/length:
  acceptance_ratio_mean max abs diff              0.0120779
  avg_cost_per_accepted_mean max abs diff         0.386336
  p95_e2e_delay_accepted_mean max abs diff        0.00158814
  qos_rejection_ratio_mean max abs diff           0.00206323

unsplit/reliability:
  acceptance_ratio_mean max abs diff              0.00255648
  avg_cost_per_accepted_mean max abs diff         0.452787
  p95_e2e_delay_accepted_mean max abs diff        0.00186477
  qos_rejection_ratio_mean max abs diff           0.00136396
```

The split-enabled results used in Figures 7--9 differ substantially from the previous plot-ready data. The reason is not random rerun noise: the previous plot-ready DMDCA rows were sourced from `paper_online_runs_split_calibrated`, whose raw DMDCA JSON files use `p_split=1.0`. The current paper setting and `run_online_paper_experiments.py` use `p_split=0.3`.

The largest differences against the old plot-ready split-enabled rows include:

```text
split_enabled/length:
  avg_cost_per_accepted_mean max abs diff         23.8958
  acceptance_ratio_mean max abs diff              0.133033

split_enabled/load:
  avg_cost_per_accepted_mean max abs diff         9.68583
  acceptance_ratio_mean max abs diff              0.0999615

split_enabled/reliability:
  avg_cost_per_accepted_mean max abs diff         19.8569
  acceptance_ratio_mean max abs diff              0.192821
```

To confirm the parameter issue, the new `split_enabled` summary was also compared against:

```text
paper_online_runs_tuned/summary_metrics.csv
```

That old tuned data uses `p_split=0.3`, and it matches the new `split_enabled` summary exactly up to floating-point formatting:

```text
split_enabled/load:        max abs diff 0 for all plotted metrics
split_enabled/length:      max abs diff 0 for all plotted metrics
split_enabled/reliability: max abs diff 0 for acceptance, cost, and p95 delay; QoS max abs diff about 1e-16
```

## Interpretation

The current data are consistent with the intended paper parameters. The visible change relative to the previous plot-ready Figures 7--9 comes from replacing the old `p_split=1.0` split-calibrated DMDCA rows with the intended `p_split=0.3` split-enabled rows.
