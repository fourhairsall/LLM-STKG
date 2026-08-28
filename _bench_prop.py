"""基准测试：修复 [2,E] 边索引 bug 后，真实全图传播的单批前反向耗时。

用于决定 batch_size / 图稀疏化策略，避免 30 epoch 跑成十几小时。
"""
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from llm_stkg.config import Config
from llm_stkg.kg.kg_builder import TourismKG
from llm_stkg.kg.llm_interface import LLMInterface
from llm_stkg.model.stkg_net import STKGNet
from llm_stkg.data.foursquare_loader import load_real_nyc
from llm_stkg.train import _build_samples, TrajDataset, _collate
from llm_stkg.head_to_head import build_ui_edge

cfg = Config()
cfg.use_bge = True
cfg.sem_dim = 768
cfg.semantic_sim_thr = 0.90
cfg.use_sgcp = True
cfg.scorer = "dot"
cfg.session_pool = "mean"
cfg.max_degree = 10
cfg.device = "cpu"

pois, checkins, test_samples, num_pois, stats, cold = load_real_nyc(None, 0.0)
sem = np.load("poi_bge_emb.npy")
kg = TourismKG(cfg, LLMInterface(bge_model_dir=None)).build(pois, checkins, sem_vecs=sem)
print("[KG] stats:", kg.stats(), flush=True)

users = list({u for u, _ in checkins})
n_users = max(users) + 1
train_samples = _build_samples(checkins, cfg.seq_len, set(users))
ui = build_ui_edge(checkins, num_pois)

model = STKGNet(cfg, num_pois, kg.num_cats, kg.cat_ids, kg.sem_vecs, kg.edge_index,
                n_users=n_users, user_item_edge=ui)
for t, ei in model.edge_index.items():
    print(f"  edge[{t}] -> {tuple(ei.shape)}", flush=True)
print("[params]", sum(p.numel() for p in model.parameters()), flush=True)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lf = nn.CrossEntropyLoss()
rng = random.Random(42)

for bs in (64, 256, 1024):
    ds = TrajDataset(train_samples, num_pois, cfg.neg_samples, rng)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=_collate)
    n_batches = (len(train_samples) + bs - 1) // bs
    it = iter(dl)
    t0 = time.time()
    k = 3
    for _ in range(k):
        H, T, C, Y, U, _x = next(it)
        sc = model(H, T, C, U)
        loss = lf(sc, Y)
        opt.zero_grad(); loss.backward(); opt.step()
    dt = (time.time() - t0) / k
    print(f"[bs={bs:5d}] {dt:.2f} s/batch | batches/epoch={n_batches} "
          f"| est epoch={dt*n_batches/60:.1f} min | est 30ep={dt*n_batches*30/3600:.2f} h",
          flush=True)
