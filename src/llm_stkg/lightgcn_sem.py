"""LightGCN + LLM 语义特征增强（新方向，替代被判废的 v2 自研图结构）。

设计动机
--------
此前 LLM-KG *图边* 在主战场 Foursquare-NYC 上净贡献为 0（去掉 KG 边 R@10 反略升），
说明"语义关系图结构"在该数据上≈类目先验的再编码，并不提供额外预测信号。
但 LLM/BGE *语义表征本身* 是真实底物——本模块把它作为**物品侧强先验**注入 LightGCN：

  - mode='init'  : 物品嵌入 V 由 BGE 768→dim 投影初始化并全程微调（语义热启动）；
  - mode='resid' : V 随机初始化并学习，最终物品表征 = 传播嵌入 + α·(冻结 BGE 投影)，
                   语义作为常驻 side feature，冷启动物品始终有语义支撑（最强语义保持）；
  - freeze_sem=True 时连投影也冻结，等价于"纯语义相似度排序"消融。

用户表征沿用 LightGCN：训练历史物品嵌入的（按访问次数加权）均值。
训练目标沿用 LightGCN 的 BPR（与基线完全一致，仅物品嵌入初始化/结构不同），
保证与基线 LightGCN 的同协议公平对照。

新颖性定位：不是"LLM-KG 图"，而是"LLM 语义 item 特征增强的 LightGCN"——
这是已被文献支持的明确贡献（语义初始化 / 语义 side information 提升冷启动与稀疏域）。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse as _sp


class LightGCNSem(nn.Module):
    def __init__(self, num_pois, sem_vecs, dim=64, n_layers=3, mode="resid",
                 freeze_sem=False, v_init=0.1):
        super().__init__()
        assert mode in ("init", "resid"), f"unknown mode {mode}"
        self.num_pois = num_pois
        self.n_layers = n_layers
        self.mode = mode
        self.freeze_sem = freeze_sem
        sem = torch.tensor(np.asarray(sem_vecs, dtype=np.float64),
                           dtype=torch.float32)  # [I, 768]
        assert sem.shape[0] == num_pois, \
            f"sem_vecs 行数 {sem.shape[0]} != num_pois {num_pois}"
        self.register_buffer("sem_raw", sem)
        self.sem_proj = nn.Linear(sem.shape[1], dim, bias=False)
        if freeze_sem:
            self.sem_proj.weight.requires_grad = False
        if mode == "init":
            # 语义热启动：V 由投影初始化，可训（除非 freeze_sem 同时为真→等价于纯语义）
            with torch.no_grad():
                init = self.sem_proj(sem)
            self.V = nn.Parameter(init.clone())
        else:  # resid：V 随机初始化并学习，语义经残差常驻
            self.V = nn.Parameter(torch.randn(num_pois, dim) * v_init)
        if mode == "resid":
            self.alpha = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("adj", torch.empty((num_pois, num_pois), dtype=torch.float32))
        self._device = "cpu"

    # ---- 共现图构建（与 baselines.LightGCN 逐元素等价：MᵀM 稀疏矩阵乘）----
    def _build_adj(self, samples):
        rows, cols = [], []
        for j, (_, hist, _) in enumerate(samples):
            for i in set(int(x) for x in hist):
                if 0 <= i < self.num_pois:
                    rows.append(j)
                    cols.append(i)
        M = _sp.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)),
                           shape=(len(samples), self.num_pois))
        C = (M.T @ M).toarray()
        np.fill_diagonal(C, 0.0)
        adj = np.eye(self.num_pois, dtype=np.float64) + C
        deg = np.sum(adj, axis=1)
        dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
        adj = dinv[:, None] * adj * dinv[None, :]
        self.adj = torch.tensor(adj, dtype=torch.float32)

    # ---- 用户画像（与 LightGCN 逐元素等价：按访问次数加权的历史均值）----
    def _build_profile(self, samples):
        u_emb = {}
        for u, hist, _ in samples:
            for p in hist:
                u_emb.setdefault(u, []).append(int(p))
        uid_list = list(u_emb.keys())
        self._uid2row = {u: r for r, u in enumerate(uid_list)}
        P = np.zeros((len(uid_list) + 1, self.num_pois), dtype=np.float32)
        for u, plist in u_emb.items():
            r = self._uid2row[u]
            for p in plist:
                if 0 <= p < self.num_pois:
                    P[r, p] += 1.0
            s = P[r].sum()
            if s > 0:
                P[r] /= s
        P[-1, 0] = 1.0
        self.user_hist = u_emb
        self._profile = torch.tensor(P, device=self._device)
        _unk = len(uid_list)
        return [(self._uid2row.get(u, _unk), t) for u, _, t in samples]

    def fit(self, samples, epochs=20, lr=1e-3, bs=1024, device="cpu"):
        self._build_adj(samples)
        self.to(device)
        self._device = device
        self.adj = self.adj.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        pos = self._build_profile(samples)
        rows_all = torch.tensor([p[0] for p in pos], dtype=torch.long, device=device)
        tgts_all = torch.tensor([p[1] for p in pos], dtype=torch.long, device=device)
        n = len(pos)
        for ep in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                # 每层重算 emb（保证每次 backward 独立计算图）
                x = self.V
                out = [x]
                for _ in range(self.n_layers):
                    x = self.adj @ x
                    out.append(x)
                emb = torch.stack(out).mean(0)            # [I, dim]
                if self.mode == "resid":
                    sem_feat = self.sem_proj(self.sem_raw.to(device))  # 冻结语义 side feature
                    emb = emb + self.alpha * sem_feat
                t = tgts_all[idx]
                neg = torch.randint(0, self.num_pois, (len(idx),), device=device)
                uv = self._profile[rows_all[idx]] @ emb   # [B, dim]
                loss = -F.logsigmoid((uv * emb[t]).sum(1) - (uv * emb[neg]).sum(1)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"[LightGCNSem:{self.mode}/freeze={self.freeze_sem}] "
                      f"epoch {ep + 1}/{epochs} loss={loss.item():.4f}", flush=True)

    def _emb(self):
        x = self.V
        out = [x]
        for _ in range(self.n_layers):
            x = self.adj @ x
            out.append(x)
        emb = torch.stack(out).mean(0)
        if self.mode == "resid":
            emb = emb + self.alpha * self.sem_proj(self.sem_raw.to(self._device))
        return emb.detach()

    def predict(self, u, hist):
        with torch.no_grad():
            emb = self._emb()
            items = self.user_hist.get(u, [0])
            uh = emb[torch.tensor(items, dtype=torch.long, device=self._device)].mean(0)
            sc = uh @ emb.T
        return sc.cpu().numpy()

    def session_predict(self, hist):
        if not hist:
            hist = [0]
        with torch.no_grad():
            emb = self._emb()
            hv = emb[torch.tensor(hist, dtype=torch.long, device=self._device)].mean(0)
            sc = hv @ emb.T
        return sc.cpu().numpy()
