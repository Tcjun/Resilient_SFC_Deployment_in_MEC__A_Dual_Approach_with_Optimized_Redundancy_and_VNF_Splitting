# Full Online Runs Data Inventory

This directory is the authoritative data root for Figures 3--9.

The top-level directory `unsplit/` is the unsplit experiment section. The four algorithms used in Figures 3--6 are stored one level deeper:

```text
paper_data/full_online_runs/unsplit/{load,length,reliability}/{Region}/{sweep_point}/{Method}/seed*/metrics_*.json
```

where `Method` is one of:

```text
MDCA
DTSP
HSPA
Random
```

For example:

```text
paper_data/full_online_runs/unsplit/load/Alabama/arrival_rate_50/DTSP/seed1/metrics_Alabama_dtsp_seed1.json
paper_data/full_online_runs/unsplit/reliability/Arizona/fixed_reliability_0p999/HSPA/seed3/metrics_Arizona_hspa_seed3.json
paper_data/full_online_runs/unsplit/length/Alabama/fixed_sfc_len_6/Random/seed5/metrics_Alabama_random_seed5.json
```

The split-enabled comparison used in Figures 7--9 is stored under the top-level `split_enabled/` directory:

```text
paper_data/full_online_runs/split_enabled/{load,length,reliability}/{Region}/{sweep_point}/{Method}/seed*/metrics_*.json
```

where `Method` is one of:

```text
MDCA
DMDCA
```

The plot script reads the consolidated summary:

```text
paper_data/full_online_runs/summary_metrics.csv
```

The raw and summary counts are:

```text
raw unsplit/load:         100 rows per method for MDCA, DTSP, HSPA, Random
raw unsplit/length:        60 rows per method for MDCA, DTSP, HSPA, Random
raw unsplit/reliability:  100 rows per method for MDCA, DTSP, HSPA, Random

raw split_enabled/load:        100 rows per method for MDCA, DMDCA
raw split_enabled/length:       60 rows per method for MDCA, DMDCA
raw split_enabled/reliability: 100 rows per method for MDCA, DMDCA

summary total: 312 rows
raw total:    1560 rows
```
