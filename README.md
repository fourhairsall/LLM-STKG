# LLM-STKG

Source code and trained model weights for **LLM-STKG: an LLM-built
Spatio-Temporal Knowledge Graph for session-based POI recommendation**
(Tang, submitted to *Knowledge-Based Systems*).

This repository accompanies the manuscript and supports the reproducibility
of the reported cross-domain POI-ranking experiments (Foursquare-NYC,
Foursquare-TKY, Gowalla, MovieLens-1M, Steam).

## Repository layout

```
LLM-STKG/
├── README.md            # this file
├── requirements.txt     # Python dependencies
├── .gitignore
├── src/                 # all source code, organised by role
│   ├── llm_stkg/        # core package: data loaders, KG builder,
│   │                   #   the STKG network, training & evaluation
│   │                   #   (data/, kg/, model/ subpackages)
│   ├── data_prep/       # raw check-in → processed graph builders
│   │                   #   (real_data_prepare*, prepare_tky_bge, gen_bge_new)
│   ├── train/           # main training drivers (p0_runs, c6_runs,
│   │                   #   run_tky_all, run_cross_new_*)
│   ├── eval_baselines/  # CF / sequential baselines & honest eval
│   │                   #   (honest_eval, esasrec/sasrec, baseline_ranks)
│   ├── generative/      # generative baseline (LLM4POI / OpenLLaMA)
│   │                   #   (llm4poi_baseline*, download_openllama,
│   │                   #   run_openllama_7b_all)
│   ├── two_stage/       # two-stage re-ranker (Appendix A negative result)
│   │                   #   (run_two_stage*)
│   └── analysis/        # diagnostics & ablation sweeps
│                       #   (hub_collapse_probe, thr_sweep, sgcp_sweep,
│                       #   cross_dataset_table, *_probe)
├── results/             # logged metric dumps referenced by the paper
│   ├── nyc/             # Foursquare-NYC main + ablation results
│   ├── tky/             # Foursquare-TKY (second geo-POI city)
│   ├── gowalla/         # Gowalla cross-domain run
│   ├── ml1m/            # MovieLens-1M cross-domain run
│   ├── steam/           # Steam cross-domain run
│   ├── cross_domain/    # Amazon-Beauty / cross-dataset head-to-head
│   ├── ablation_graph/  # KG-ablation variants (cat/covisit/geo/sem,
│   │                   #   depth1-2, t_thr_*, thr_sweep, sgcp_sweep)
│   ├── two_stage/       # two-stage re-ranker negative-result dumps
│   └── generative/      # LLM4POI / SASRec / honest-eval baseline dumps
└── weights/             # trained .pt checkpoints (see below)
```

> **Note.** All experiment scripts import the `llm_stkg` package, so run them
> with `src/` on the path (e.g. `PYTHONPATH=src`) or from inside `src/`.
> Root-level exploratory/debug artifacts have been removed from version
> control; they remain recoverable from git history if needed.

## Environment

- Python 3.10+
- PyTorch 2.x with CUDA 11.8+ (CPU-only also runs, slower)
- See `requirements.txt` for the full dependency list

```bash
pip install -r requirements.txt
```

## Data

The Foursquare-NYC and Foursquare-TKY check-in datasets are publicly
released mobility datasets (Dingqi Yang, Daqing Zhang, et al., UMAP 2014/2015);
original source: <https://sites.google.com/site/yangdingqi/home/foursquare-dataset>.

The semantic encoder is the public `BAAI/bge-base-en-v1.5` model on
HuggingFace and is loaded frozen.

Build the processed graphs from the raw check-ins with:

```bash
PYTHONPATH=src python src/llm_stkg/real_data_prepare.py        # Foursquare-NYC
PYTHONPATH=src python src/data_prep/real_data_prepare_tky.py   # Foursquare-TKY
```

## Reproducing the results

1. Prepare the datasets as above.
2. Train / evaluate via `src/llm_stkg/run_experiment.py` and the drivers under
   `src/train/`, `src/eval_baselines/`, `src/generative/`, `src/two_stage/`
   (see the filenames for each baseline and ablation).
3. To reuse the reported checkpoints directly, load the provided `weights/*.pt`
   files instead of retraining.

### Trained weights

| File | Description |
|------|-------------|
| `weights/nyc_c6u_seed42.pt` | NYC main model (C6 + SGCP, seed 42) |
| `weights/nyc_nosgcp_s42.pt`  | NYC ablation (no SGCP) |
| `weights/tky_c6u_seed42.pt`  | TKY main model (seed 42) |

## Results index (paper ↔ repository)

- **Foursquare-NYC main + ablations** → `results/nyc/`
- **Foursquare-TKY (second city)** → `results/tky/`
- **Cross-domain (Gowalla / MovieLens-1M / Steam)** → `results/gowalla/`,
  `results/ml1m/`, `results/steam/`
- **KG-ablation variants** → `results/ablation_graph/`
- **Two-stage re-ranker (Appendix A negative result)** → `results/two_stage/`
- **Generative / sequential baselines** → `results/generative/`

## Citation

If you use this code or the reported results, please cite the accompanying
paper:

```bibtex
@article{tang2026llmstkg,
  title  = {LLM-STKG: An LLM-built Spatio-Temporal Knowledge Graph for Session-based POI Recommendation},
  author = {Tang, Shujiang},
  journal = {Knowledge-Based Systems},
  year   = {2026},
  note   = {submitted}
}
```

## License

MIT
