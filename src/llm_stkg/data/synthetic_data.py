"""合成数据生成器：用于无网络/无下载时跑通整条流水线（smoke test）。

生成具有「同类别聚集 + 地理邻近」结构的签到轨迹，便于验证模型能学到时空-语义信号。
真实实验请改用 data/foursquare_loader.py 加载 Foursquare NYC/TKY 或 LLM4POI 预处理集。
"""
import random


def generate_synthetic(cfg, seed: int = 42):
    rng = random.Random(seed)
    # 每个类别一段专属主题词，使同类 POI 文本相似、异类不相似（语义边稀疏且有意义）
    cat_words = [
        "museum art history exhibition gallery painting",
        "cafe coffee espresso bakery pastry breakfast",
        "restaurant dinner cuisine dining seafood steak",
        "park nature outdoor hiking lake mountain",
        "shopping mall retail store fashion clothes",
        "hotel lodging accommodation room suite",
        "cinema movie theater film comedy",
        "gym fitness sport workout training",
        "bar pub nightlife music beer",
        "library book reading study quiet",
    ]
    pois = []
    for pid in range(cfg.num_pois):
        cat = pid % cfg.num_categories
        lat = 40.0 + rng.uniform(-0.15, 0.15) + rng.gauss(0, 0.02)
        lng = 116.0 + rng.uniform(-0.15, 0.15) + rng.gauss(0, 0.02)
        text = cat_words[cat]
        pois.append({"poi_id": pid, "category": cat, "lat": lat, "lng": lng, "text": text})

    checkins = []
    for uid in range(cfg.num_users):
        # 每个用户有强偏好：以 0.85 概率留在同类别，且偏向邻近 POI
        cur = rng.randint(0, cfg.num_pois - 1)
        seq = []
        for _ in range(cfg.seq_len):
            same = [p for p in range(cfg.num_pois)
                    if p != cur and pois[p]["category"] == pois[cur]["category"]]
            if same and rng.random() < 0.85:
                # 在同类别里挑地理最近的若干个之一，制造可学习的时空信号
                same.sort(key=lambda p: (pois[p]["lat"] - pois[cur]["lat"]) ** 2
                                       + (pois[p]["lng"] - pois[cur]["lng"]) ** 2)
                cur = rng.choice(same[: max(1, len(same) // 5)])
            else:
                cur = rng.randint(0, cfg.num_pois - 1)
            seq.append(cur)
        checkins.append((uid, seq))
    return pois, checkins
