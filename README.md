# LLM-STKG

Source code and trained model weights for **LLM-STKG: an LLM-built
Spatio-Temporal Knowledge Graph for session-based POI recommendation**
(Tang, submitted to *Knowledge-Based Systems*).

This repository accompanies the manuscript and supports the reproducibility
of the reported cross-domain POI-ranking experiments (Foursquare-NYC,
Foursquare-TKY, Gowalla, MovieLens-1M, Steam).

## Repository layout

- `llm_stkg/` — the main package: data loaders, KG builder, the STKG
  network, training and evaluation routines.
- `*.py` (repository root) — experiment drivers, baseline rankers, and
  diagnostic probes used to produce the tables/figures in the paper.
- `*.pt` — trained model weights for the reported NYC / TKY results and
  their ablation variants.
- `*.json` — logged metric dumps referenced by the result tables.

## Environment

- Python 3.10+
- PyTorch 2.x with CUDA 11.8+ (CPU-only also runs, slower)
- See `requirements.txt` for the full dependency list

Install with:

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
python llm_stkg/real_data_prepare.py        # Foursquare-NYC
python real_data_prepare_tky.py             # Foursquare-TKY
```

## Reproducing the results

1. Prepare the datasets as above.
2. Train / evaluate via `llm_stkg/run_experiment.py` and the drivers in the
   repository root (see the filenames for each baseline and ablation).
3. To reuse the reported checkpoints directly, load the provided `*.pt`
   weights (paths listed below) instead of retraining.

### Trained weights

| File | Description |
|------|-------------|
| `_c6u_seed42.pt` | NYC main model (seed 42) |
| `tky_c6u_seed42.pt` | TKY main model (seed 42) |
| `nyc_nosgcp_s42.pt` | NYC ablation (no SGCP) |
| `_diag_attn.pt` | diagnostic attention weights |

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
