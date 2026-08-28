"""出行意图推理模块 (C3)：把用户轨迹的语义/时空信号聚合为意图向量。

真实部署：可接入 LLM 因式分解提示（如 KAR 的 factorization prompting）对用户出行意图做
推理，再对齐到推荐模型；本地回退用轨迹 POI 语义均值 + 可学习投影，保证可离线复现。
"""
import torch
import torch.nn as nn


class IntentModule(nn.Module):
    def __init__(self, sem_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(sem_dim, hidden_dim)

    def forward(self, traj_sem_emb: torch.Tensor):
        # traj_sem_emb: [B, L, sem_dim]
        mask = (traj_sem_emb.abs().sum(-1) > 0).float()  # [B, L]
        denom = mask.sum(1, keepdim=True).clamp_min(1.0)
        pooled = traj_sem_emb.sum(1) / denom
        return torch.tanh(self.proj(pooled))
