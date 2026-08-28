"""烟囱测试：用合成数据跑通 LLM-STKG 整条流水线（无需下载/无需 GPU）。

运行：
  python run_smoke.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_stkg.config import Config
from llm_stkg.data.synthetic_data import generate_synthetic
from llm_stkg.train import train_model


def main():
    cfg = Config(
        num_users=200, num_pois=500, num_categories=10, seq_len=20,
        epochs=15, batch_size=64, neg_samples=10, device="cpu",
    )
    print("[Config]\n" + str(cfg))
    print("[Data] 生成合成签到数据 ...")
    pois, checkins = generate_synthetic(cfg, seed=cfg.seed)
    print(f"  POIs={len(pois)} Users={len(checkins)}")

    model, metrics = train_model(cfg, pois, checkins)
    print("\n[Smoke Test 通过] 模型可训练、可评估。验证集指标:")
    for k, v in metrics.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
