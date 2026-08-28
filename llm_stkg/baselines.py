"""可跑基线 (同协议对比)：BPR-MF / FPMC / LightGCN / GRU-STGN-lite。

全部在 GPU 上训练、全候选排名评估，与 LLM-STKG 共用 prepare_splits 的划分，保证公平。
- BPR-MF      : 非序列化矩阵分解（经典强基线）
- FPMC        : 序列化因式分解（用户因子 + 物品因子 + 转移因子）
- LightGCN    : 基于共现物品图的轻量图卷积（无时序）
- GRU-STGN    : 纯序列化 GRU（无 KG / 无 LLM），用于隔离「KG+LLM」的独立贡献
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _full_eval(predict_fn, val_samples, num_pois, k_list=(5, 10), device="cpu"):
    """predict_fn(user, hist) -> np.array[num_pois]；全候选排名。"""
    from .evaluate import rank_metrics
    all_scores, all_tgt = [], []
    for u, hist, tgt in val_samples:
        sc = torch.tensor(predict_fn(u, hist), dtype=torch.float32)
        all_scores.append(sc)
        all_tgt.append(tgt)
    scores = torch.stack(all_scores)
    tgts = torch.tensor(all_tgt, dtype=torch.long)
    return rank_metrics(scores, tgts, k_list=k_list)


class BPRMF(nn.Module):
    def __init__(self, num_users, num_pois, dim=64):
        super().__init__()
        self.U = nn.Embedding(num_users, dim)
        self.V = nn.Embedding(num_pois, dim)
        self.num_pois = num_pois
        self._device = "cpu"

    def fit(self, samples, epochs=20, lr=1e-3, bs=1024, device="cpu"):
        self.to(device)
        self._device = device
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        pos = [(u, t) for u, _, t in samples]
        for ep in range(epochs):
            random.shuffle(pos)
            for i in range(0, len(pos), bs):
                batch = pos[i:i + bs]
                u = torch.tensor([x[0] for x in batch], device=device)
                t = torch.tensor([x[1] for x in batch], device=device)
                neg = torch.randint(0, self.num_pois, (len(batch),), device=device)
                pu = self.U(u); pv = self.V(t); nv = self.V(neg)
                loss = -F.logsigmoid((pu * pv).sum(1) - (pu * nv).sum(1)).mean()
                opt.zero_grad(); loss.backward(); opt.step()

    def predict(self, u, hist):
        with torch.no_grad():
            sc = (self.U(torch.tensor(u, device=self._device)).detach() @ self.V.weight.detach().T)
        return sc.cpu().numpy()

    def session_predict(self, hist):
        """跨用户测试公平版：用历史物品嵌入均值作用户表征，避免 test 用户无 embedding。"""
        with torch.no_grad():
            if not hist:
                hist = [0]
            hv = self.V(torch.tensor(hist, dtype=torch.long, device=self._device)).mean(0)
            sc = hv @ self.V.weight.T
        return sc.cpu().numpy()

    def eval_metrics(self, val_samples, device="cpu"):
        return _full_eval(lambda u, h: self.predict(u, h), val_samples, self.num_pois, device=device)


class FPMC(nn.Module):
    def __init__(self, num_users, num_pois, dim=64):
        super().__init__()
        self.U = nn.Embedding(num_users, dim)
        self.V = nn.Embedding(num_pois, dim)
        self.T = nn.Embedding(num_pois, dim)
        self.num_pois = num_pois
        self._device = "cpu"

    def fit(self, samples, epochs=20, lr=1e-3, bs=1024, device="cpu"):
        self.to(device)
        self._device = device
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        triples = [(u, h[-1] if h else 0, t) for u, h, t in samples]
        for ep in range(epochs):
            random.shuffle(triples)
            for i in range(0, len(triples), bs):
                batch = triples[i:i + bs]
                u = torch.tensor([x[0] for x in batch], device=device)
                last = torch.tensor([x[1] for x in batch], device=device)
                t = torch.tensor([x[2] for x in batch], device=device)
                neg = torch.randint(0, self.num_pois, (len(batch),), device=device)
                pos_sc = ((self.U(u) + self.T(last)) * self.V(t)).sum(1)
                neg_sc = ((self.U(u) + self.T(last)) * self.V(neg)).sum(1)
                loss = -F.logsigmoid(pos_sc - neg_sc).mean()
                opt.zero_grad(); loss.backward(); opt.step()

    def predict(self, u, hist):
        last = hist[-1] if hist else 0
        with torch.no_grad():
            ctx = (self.U(torch.tensor(u, device=self._device)) + self.T(torch.tensor(last, device=self._device))).detach()
            sc = ctx @ self.V.weight.detach().T
        return sc.cpu().numpy()

    def session_predict(self, hist):
        with torch.no_grad():
            if not hist:
                hist = [0]
            last = hist[-1]
            hv = self.V(torch.tensor(hist, dtype=torch.long, device=self._device)).mean(0)
            ctx = hv + self.T(torch.tensor(last, device=self._device))
            sc = ctx @ self.V.weight.T
        return sc.cpu().numpy()

    def eval_metrics(self, val_samples, device="cpu"):
        return _full_eval(lambda u, h: self.predict(u, h), val_samples, self.num_pois, device=device)


class LightGCN(nn.Module):
    def __init__(self, num_pois, dim=64, n_layers=3):
        super().__init__()
        self.V = nn.Embedding(num_pois, dim)
        self.num_pois = num_pois
        self.n_layers = n_layers
        self.register_buffer("ei", torch.empty((2, 0), dtype=torch.long))
        self._device = "cpu"

    def _build_adj(self, samples):
        # 由训练样本共现构建物品-物品对称图（稠密，I<=~3000 适用）
        #
        # 【性能】原实现用 Python 双重循环枚举每条历史内的所有物品对，代价是
        # Σ_s |hist_s|²/2 次解释器迭代：Steam（188k 样本 × 均长 53.6）约 2.7e8 次、
        # Gowalla（168k × 81.7）约 5.6e8 次，实测卡死 20 分钟以上。
        # 改为稀疏指示矩阵一次矩阵乘：M[j, i]=1 表示样本 j 的历史含物品 i，
        # 则 C = MᵀM 的元素 C[i][j] 恰为"同时包含 i 与 j 的样本数"，与原双重循环
        # 对 adj[i][j] 的累加量【逐元素相等】。对角线不累加（保持 np.eye 的 1.0），
        # 与原实现一致（原代码只遍历 a<b，从不触及对角线）。数值等价，仅耗时不同。
        from scipy import sparse as _sp
        rows, cols = [], []
        for j, (_, hist, _) in enumerate(samples):
            for i in set(int(x) for x in hist):
                if 0 <= i < self.num_pois:
                    rows.append(j)
                    cols.append(i)
        M = _sp.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)),
                           shape=(len(samples), self.num_pois))
        C = (M.T @ M).toarray()
        np.fill_diagonal(C, 0.0)          # 对角线由下面的 eye 提供，避免重复计数
        adj = np.eye(self.num_pois, dtype=np.float64) + C
        deg = np.sum(adj, axis=1)
        dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
        adj = dinv[:, None] * adj * dinv[None, :]
        self.adj = torch.tensor(adj, dtype=torch.float32)

    def fit(self, samples, epochs=20, lr=1e-3, bs=1024, device="cpu"):
        self._build_adj(samples)
        self.to(device)
        self._device = device
        self.adj = self.adj.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        # 用户表征 = 训练历史物品嵌入均值
        u_emb = {}
        for u, hist, _ in samples:
            for p in hist:
                u_emb.setdefault(u, []).append(int(p))
        self.user_hist = u_emb
        # 【性能】用户表征原按 batch 逐样本 Python 循环建 tensor 再 mean（每 batch 1024 次），
        # 改为预计算稠密画像矩阵 P：P[r, i] = 用户在历史中访问物品 i 的次数 / 历史总长，
        # 于是 uv = P[rows] @ emb 与原来的 emb[hist].mean(0) 【逐元素相等】（同为按出现
        # 次数加权的均值）。只为实际出现过的用户建行，避免 n_users 取 max(uid)+1 时爆内存。
        uid_list = list(u_emb.keys())
        self._uid2row = {u: r for r, u in enumerate(uid_list)}
        P = np.zeros((len(uid_list) + 1, self.num_pois), dtype=np.float32)  # 末行=未知用户
        for u, plist in u_emb.items():
            r = self._uid2row[u]
            for p in plist:
                if 0 <= p < self.num_pois:
                    P[r, p] += 1.0
            s = P[r].sum()
            if s > 0:
                P[r] /= s
        P[-1, 0] = 1.0                       # 未知用户回退到原实现的 [0] 占位行为
        self._profile = torch.tensor(P, device=device)
        _unk = len(uid_list)
        pos = [(self._uid2row.get(u, _unk), t) for u, _, t in samples]
        rows_all = torch.tensor([p[0] for p in pos], dtype=torch.long, device=device)
        tgts_all = torch.tensor([p[1] for p in pos], dtype=torch.long, device=device)
        n = len(pos)
        for ep in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                # 每层重算 emb，保证每次 backward 使用独立计算图
                x = self.V.weight
                out = [x]
                for _ in range(self.n_layers):
                    x = self.adj @ x
                    out.append(x)
                emb = torch.stack(out).mean(0)          # [I, dim]
                t = tgts_all[idx]
                neg = torch.randint(0, self.num_pois, (len(idx),), device=device)
                uv = self._profile[rows_all[idx]] @ emb  # [B, dim]
                loss = -F.logsigmoid((uv * emb[t]).sum(1) - (uv * emb[neg]).sum(1)).mean()
                opt.zero_grad(); loss.backward(); opt.step()

    def _emb(self):
        x = self.V.weight
        out = [x]
        for _ in range(self.n_layers):
            x = self.adj @ x
            out.append(x)
        return torch.stack(out).mean(0).detach()

    def predict(self, u, hist):
        with torch.no_grad():
            emb = self._emb()
            items = self.user_hist.get(u, [0])
            uh = emb[torch.tensor(items, dtype=torch.long, device=self._device)].mean(0)
            sc = uh @ emb.T
        return sc.cpu().numpy()

    def session_predict(self, hist):
        with torch.no_grad():
            emb = self._emb()
            if not hist:
                hist = [0]
            hv = emb[torch.tensor(hist, dtype=torch.long, device=self._device)].mean(0)
            sc = hv @ emb.T
        return sc.cpu().numpy()

    def eval_metrics(self, val_samples, device="cpu"):
        return _full_eval(lambda u, h: self.predict(u, h), val_samples, self.num_pois, device=device)


class GRU_STGN(nn.Module):
    """纯序列化 GRU（无 KG / 无 LLM），作为 ours 的对照基线。"""

    def __init__(self, num_users, num_pois, dim=64):
        super().__init__()
        self.item_emb = nn.Embedding(num_pois, dim)
        self.user_emb = nn.Embedding(num_users, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.num_pois = num_pois
        self._device = "cpu"

    def fit(self, samples, epochs=20, lr=1e-3, bs=256, device="cpu"):
        self.to(device)
        self._device = device
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for ep in range(epochs):
            random.shuffle(samples)
            for i in range(0, len(samples), bs):
                batch = samples[i:i + bs]
                maxl = max(len(h) for _, h, _ in batch)
                H = torch.zeros(len(batch), maxl, device=device, dtype=torch.long)
                for bi, (_, h, _) in enumerate(batch):
                    H[bi, :len(h)] = torch.tensor(h, dtype=torch.long, device=device)
                U = torch.tensor([u for u, _, _ in batch], device=device)
                T = torch.tensor([t for _, _, t in batch], device=device)
                he = self.item_emb(H)
                _, hn = self.gru(he)
                uh = hn.squeeze(0) + self.user_emb(U)
                sc = uh @ self.item_emb.weight.T
                loss = loss_fn(sc, T)
                opt.zero_grad(); loss.backward(); opt.step()

    def predict(self, u, hist):
        if not hist:
            hist = [0]
        with torch.no_grad():
            H = torch.tensor([hist], dtype=torch.long, device=self._device)
            he = self.item_emb(H)
            _, hn = self.gru(he)
            uh = hn.squeeze(0) + self.user_emb(torch.tensor([u], device=self._device))
            sc = uh @ self.item_emb.weight.T
        return sc.squeeze(0).cpu().numpy()

    def session_predict(self, hist):
        if not hist:
            hist = [0]
        with torch.no_grad():
            H = torch.tensor([hist], dtype=torch.long, device=self._device)
            he = self.item_emb(H)
            _, hn = self.gru(he)
            uh = hn.squeeze(0)
            sc = uh @ self.item_emb.weight.T
        return sc.squeeze(0).cpu().numpy()

    def eval_metrics(self, val_samples, device="cpu"):
        return _full_eval(lambda u, h: self.predict(u, h), val_samples, self.num_pois, device=device)


class SASRec(nn.Module):
    """自注意力序列模型 (Kang & McAuley, ICDM'18)，作为 ours 的强序列对照基线。

    与 GRU-STGN 的差别仅在序列编码器：用多头自注意力（因果掩码）替代 GRU 最后隐状态，
    因此可直接关注历史中任意位置（含最近一次访问的 POI），缓解长序列梯度消失。
    公平协议：同样以 (hist -> tgt) 的下一物品交叉熵训练，同 session_predict 全候选打分。
    """

    def __init__(self, num_users, num_pois, dim=64, n_layers=2, n_heads=2,
                 maxlen=200, dropout=0.1):
        super().__init__()
        # 预留索引 num_pois 作为 PAD（左补齐用），避免与真实 POI 0 冲突
        self.item_emb = nn.Embedding(num_pois + 1, dim)
        self.pos_emb = nn.Embedding(maxlen, dim)
        self.pad = num_pois
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.maxlen = maxlen
        self.drop = nn.Dropout(dropout)
        self.attn = nn.ModuleList(
            [nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
             for _ in range(n_layers)])
        self.attn_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(dim * 4, dim))
            for _ in range(n_layers)])
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.num_pois = num_pois
        self._device = "cpu"
        self.dim = dim

    def _forward(self, seq, key_padding_mask=None):
        """seq: [B, L] (含 PAD)；返回 [B, L, dim]。"""
        L = seq.size(1)
        pos = torch.arange(L, device=seq.device).unsqueeze(0)            # [1, L]
        x = self.item_emb(seq) + self.pos_emb(pos)                       # [B, L, dim]
        x = self.drop(x)
        causal = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool),
                            diagonal=1)                                  # True=未来，被掩
        for i in range(self.n_layers):
            h, _ = self.attn[i](x, x, x, attn_mask=causal,
                                key_padding_mask=key_padding_mask, need_weights=False)
            h = self.drop(h)
            x = self.attn_norm[i](x + h)                                 # Post-LN
            f = self.ffn[i](x)
            f = self.drop(f)
            x = self.ffn_norm[i](x + f)
        return x

    def fit(self, samples, epochs=30, lr=1e-3, bs=256, device="cpu",
            max_train=None):
        self.to(device)
        self._device = device
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad)
        if max_train:
            samples = samples[:max_train]
        Lmax = min(max(len(h) for _, h, _ in samples), self.maxlen)
        for ep in range(epochs):
            random.shuffle(samples)
            for i in range(0, len(samples), bs):
                batch = samples[i:i + bs]
                L = min(max(len(h) for _, h, _ in batch), Lmax)
                idx = torch.full((len(batch), L), self.pad,
                                 dtype=torch.long, device=device)
                T = torch.zeros(len(batch), dtype=torch.long, device=device)
                kpm = torch.zeros(len(batch), L, dtype=torch.bool, device=device)
                for bi, (_, h, t) in enumerate(batch):
                    h = h[-L:]
                    idx[bi, L - len(h):] = torch.tensor(h, dtype=torch.long, device=device)
                    kpm[bi, :L - len(h)] = True                         # 左补齐处为 PAD
                    T[bi] = int(t)
                x = self._forward(idx, key_padding_mask=kpm)            # [B, L, dim]
                # 左补齐 → 最右列 L-1 恒为真实最后一项（h[-L:] 的末位）
                last = x[:, -1, :]                                       # [B, dim]
                sc = last @ self.item_emb.weight[:self.num_pois].T       # [B, num_pois]
                loss = loss_fn(sc, T)
                opt.zero_grad(); loss.backward(); opt.step()
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"[SASRec] epoch {ep + 1}/{epochs} loss={loss.item():.4f}",
                      flush=True)

    def session_predict(self, hist):
        if not hist:
            hist = [0]
        h = hist[-self.maxlen:]
        with torch.no_grad():
            seq = torch.tensor([h], dtype=torch.long, device=self._device)
            x = self._forward(seq)
            last = x[:, -1, :]
            sc = last @ self.item_emb.weight[:self.num_pois].T
        return sc.squeeze(0).cpu().numpy()

    def eval_metrics(self, val_samples, device="cpu"):
        return _full_eval(lambda u, h: self.session_predict(h), val_samples,
                          self.num_pois, device=device)


class RMSNorm(nn.Module):
    """Root-mean-square 归一化（LiGR / Llama 风格）。"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: [..., dim]
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU 前馈（LiGR / Llama 风格）。"""
    def __init__(self, dim, hidden, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.drop(self.w2(self.act(self.w1(x)) * self.w3(x)))


class LiGRBlock(nn.Module):
    """LiGR (Llama for Generative Recommendation) 块：Pre-LN(RMSNorm) + 因果注意力 + SwiGLU。"""
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, dim * 4, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        nx = self.norm1(x)
        h, _ = self.attn(nx, nx, nx, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop(h)
        x = x + self.ffn(self.norm2(x))
        return x


class eSASRec(nn.Module):
    """eSASRec (Tikhonovich et al., RecSys'25)：SASRec 训练目标 + LiGR Transformer 层
    (RMSNorm + SwiGLU，Llama 风格) + Sampled Softmax Loss。

    与 SASRec 的唯一差别在 (1) 序列块改用 LiGR，(2) 训练损失改用 sampled softmax
    （in-batch 负样本 + 均匀随机负样本，而非对全 4980 个物品做 full softmax）。
    前向与 session_predict 与 SASRec 完全一致（同为左补齐 + 最右列取末位 + 点积打分），
    以保证只隔离前述两个变量的差异，公平回答「2025 SOTA 序列增强在 replay 主导数据上能否
    超越 SASRec、能否逼近 replay+行为先验（ours）」。
    """

    def __init__(self, num_users, num_pois, dim=64, n_layers=2, n_heads=2,
                 maxlen=200, dropout=0.1, num_neg=100, loss_mode="sampled_softmax"):
        super().__init__()
        self.item_emb = nn.Embedding(num_pois + 1, dim)   # 预留索引 num_pois 作 PAD
        self.pos_emb = nn.Embedding(maxlen, dim)
        self.pad = num_pois
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.maxlen = maxlen
        self.num_neg = num_neg
        self.loss_mode = loss_mode
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [LiGRBlock(dim, n_heads, dropout) for _ in range(n_layers)])
        self.num_pois = num_pois
        self._device = "cpu"
        self.dim = dim

    def _forward(self, seq, key_padding_mask=None):
        L = seq.size(1)
        pos = torch.arange(L, device=seq.device).unsqueeze(0)
        x = self.item_emb(seq) + self.pos_emb(pos)
        x = self.drop(x)
        causal = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool),
                            diagonal=1)
        for block in self.blocks:
            x = block(x, attn_mask=causal, key_padding_mask=key_padding_mask)
        return x

    def fit(self, samples, epochs=30, lr=1e-3, bs=256, device="cpu",
            max_train=None):
        self.to(device)
        self._device = device
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        if max_train:
            samples = samples[:max_train]
        Lmax = min(max(len(h) for _, h, _ in samples), self.maxlen)
        for ep in range(epochs):
            random.shuffle(samples)
            for i in range(0, len(samples), bs):
                batch = samples[i:i + bs]
                L = min(max(len(h) for _, h, _ in batch), Lmax)
                idx = torch.full((len(batch), L), self.pad,
                                 dtype=torch.long, device=device)
                T = torch.zeros(len(batch), dtype=torch.long, device=device)
                kpm = torch.zeros(len(batch), L, dtype=torch.bool, device=device)
                for bi, (_, h, t) in enumerate(batch):
                    h = h[-L:]
                    idx[bi, L - len(h):] = torch.tensor(h, dtype=torch.long, device=device)
                    kpm[bi, :L - len(h)] = True
                    T[bi] = int(t)
                x = self._forward(idx, key_padding_mask=kpm)        # [B, L, dim]
                last = x[:, -1, :]                                   # [B, dim]
                if self.loss_mode == "ce":
                    # 与 SASRec 完全相同：对全 POI 集合的 plain cross-entropy（均匀负样本）
                    sc = last @ self.item_emb.weight[:self.num_pois].T
                    loss = F.cross_entropy(sc, T)
                else:
                    # ---- Sampled Softmax Loss（eSASRec 原版）----
                    pos_emb = self.item_emb(T)                      # [B, dim]
                    s_pos = (last * pos_emb).sum(1)                 # [B]
                    S = last @ self.item_emb(T).T                   # [B, B]
                    eye = torch.eye(len(batch), dtype=torch.bool, device=device)
                    neg_ib = S[~eye].view(len(batch), len(batch) - 1)  # [B, B-1]
                    rand = torch.randint(0, self.num_pois,
                                         (len(batch), self.num_neg), device=device)
                    neg_rand = torch.einsum("bd,bnd->bn", last,
                                            self.item_emb(rand))    # [B, num_neg]
                    neg = torch.cat([neg_ib, neg_rand], dim=1)      # [B, B-1+num_neg]
                    logits = torch.cat([s_pos.unsqueeze(1), neg], dim=1)
                    loss = F.cross_entropy(
                        logits, torch.zeros(len(batch), dtype=torch.long, device=device))
                opt.zero_grad(); loss.backward(); opt.step()
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"[eSASRec:{self.loss_mode}] epoch {ep + 1}/{epochs} "
                      f"loss={loss.item():.4f}", flush=True)

    def session_predict(self, hist):
        if not hist:
            hist = [0]
        h = hist[-self.maxlen:]
        with torch.no_grad():
            seq = torch.tensor([h], dtype=torch.long, device=self._device)
            x = self._forward(seq)
            last = x[:, -1, :]
            sc = last @ self.item_emb.weight[:self.num_pois].T
        return sc.squeeze(0).cpu().numpy()

    def eval_metrics(self, val_samples, device="cpu"):
        return _full_eval(lambda u, h: self.session_predict(h), val_samples,
                          self.num_pois, device=device)


def build_baselines(num_users, num_pois, device="cpu", with_sasrec=True):
    out = {
        "BPR-MF": BPRMF(num_users, num_pois),
        "FPMC": FPMC(num_users, num_pois),
        "LightGCN": LightGCN(num_pois),
        "GRU-STGN": GRU_STGN(num_users, num_pois),
    }
    if with_sasrec:
        out["SASRec"] = SASRec(num_users, num_pois)
    return out
