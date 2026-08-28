"""真实数据加载器：兼容 Foursquare / Gowalla / LLM4POI 预处理 CSV 格式。

期望列（至少）：UserId, PoiId, Latitude, Longitude, PoiCategoryId, PoiCategoryName,
UTCTimeOffsetEpoch（用于按时序切分轨迹）。
"""
import csv
from collections import defaultdict


def load_foursquare(path, max_users=None, max_pois=None):
    user_map, poi_map = {}, {}
    raw = defaultdict(list)
    poi_meta_tmp = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            u = row.get("UserId") or row.get("user_id")
            p = row.get("PoiId") or row.get("poi_id")
            if u is None or p is None:
                continue
            if u not in user_map:
                user_map[u] = len(user_map)
            if p not in poi_map:
                poi_map[p] = len(poi_map)
                poi_meta_tmp[poi_map[p]] = {
                    "poi_id": poi_map[p],
                    "category": int(float(row.get("PoiCategoryId", 0) or 0)),
                    "lat": float(row.get("Latitude", 0)),
                    "lng": float(row.get("Longitude", 0)),
                    "text": f"{row.get('PoiCategoryName','')} {row.get('PoiCategoryId','')}",
                }
            try:
                ts = float(row.get("UTCTimeOffsetEpoch", 0) or 0)
            except ValueError:
                ts = 0.0
            raw[user_map[u]].append((ts, poi_map[p]))
            if max_users and len(user_map) >= max_users:
                break
    checkins = []
    for u, lst in raw.items():
        lst.sort()
        seq = [p for _, p in lst]
        checkins.append((u, seq))
    pois = [poi_meta_tmp[i] for i in range(len(poi_meta_tmp))]
    return pois, checkins


# ---------- Foursquare-NYC 同构替代数据生成器 ----------
# 说明：本沙箱无法下载 HuggingFace / archive.org / Google Drive 上的真实 LLM4POI 数据，
# 故生成「schema 与 Foursquare-NYC/LLM4POI 完全一致」的替代数据用于验证 GPU 全流程。
# 字段、轨迹结构、时空-语义信号均对齐真实数据；真实数据到位后用 load_foursquare 一行换入。
CAT_NAMES = [
    "Airport", "Coffee Shop", "Restaurant", "Bar", "Museum", "Park", "Hotel",
    "Shopping Mall", "Gym", "Library", "Theater", "Stadium", "Zoo", "Beach",
    "Art Gallery", "Bakery", "Nightclub", "Pharmacy", "Bank", "School",
]
REGION_NAMES = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island",
                "Jersey City", "Hoboken", "Newark"]


def generate_foursquare_like(num_users=4000, num_pois=1500, seq_min=10, seq_max=30,
                             seed=42, city_center=(40.73, -73.93), revisit_ratio=0.5):
    """生成具有「区域聚集 + 用户类别偏好 + 偶发跨区探索 + 可控重访率」结构的 NYC 风格签到序列。

    revisit_ratio: 每一步以该概率重访序列中最近一次访问的 POI（模拟用户回访行为）；
                   =0 时纯探索，=1 时纯重访。用于受控验证「HF/历史频率基线强势
                   是否依赖高重访率」（替代不可行的 TKY 跨城市验证）。
    """
    import random
    rng = random.Random(seed)
    # 8 个区域中心围绕城市中心散布
    regions = [(city_center[0] + rng.uniform(-0.12, 0.12),
                city_center[1] + rng.uniform(-0.12, 0.12)) for _ in range(len(REGION_NAMES))]
    pois = []
    poi_region = []
    for pid in range(num_pois):
        r = rng.randrange(len(regions))
        cat = rng.randrange(len(CAT_NAMES))
        lat = regions[r][0] + rng.gauss(0, 0.02)
        lng = regions[r][1] + rng.gauss(0, 0.02)
        text = f"{CAT_NAMES[cat]} {REGION_NAMES[r]}"
        pois.append({"poi_id": pid, "category": cat, "lat": lat, "lng": lng, "text": text})
        poi_region.append(r)

    # 每 POI 预计算所属区域，便于按区域采样
    region_pois = {}
    for pid, r in enumerate(poi_region):
        region_pois.setdefault(r, []).append(pid)

    checkins = []
    for uid in range(num_users):
        home = rng.randrange(len(regions))
        # 用户类别偏好（前几类权重高）
        aff = list(range(len(CAT_NAMES)))
        rng.shuffle(aff)
        pref = {c: (len(CAT_NAMES) - i) for i, c in enumerate(aff)}
        L = rng.randint(seq_min, seq_max)
        seq = []
        cur = rng.choice(region_pois[home])
        for _ in range(L):
            if seq and rng.random() < revisit_ratio:
                nxt = seq[-1]  # 重访最近一次访问的 POI
            else:
                if rng.random() < 0.82:
                    # 留在本区域，按偏好挑类别
                    cand = region_pois[home]
                else:
                    # 跨区探索（游客行为）
                    cand = region_pois[rng.randrange(len(regions))]
                # 在候选里按类别偏好加权选
                weights = [pref[pois[p]["category"]] for p in cand]
                nxt = rng.choices(cand, weights=weights, k=1)[0]
            seq.append(nxt)
        checkins.append((uid, seq))
    return pois, checkins


def load_or_generate(path, cfg, max_pois=None):
    """优先加载真实 Foursquare/LLM4POI CSV；不存在则生成同构替代数据。返回 (pois, checkins, source)。"""
    import os
    if path and os.path.exists(path):
        pois, checkins = load_foursquare(path)
        source = f"real:{path}"
    else:
        pois, checkins = generate_foursquare_like(
            num_users=cfg.num_users, num_pois=cfg.num_pois, seed=cfg.seed)
        source = "surrogate:Foursquare-NYC-schema"
    if max_pois:
        # 抽取高频 POI 子集，控制图规模（真实数据规模大时必做）
        from collections import Counter
        cnt = Counter(p for _, seq in checkins for p in seq)
        keep = set(i for i, _ in cnt.most_common(max_pois))
        new_id = {p: k for k, p in enumerate(keep)}
        pois = [pois[p] for p in keep]
        for i, m in enumerate(pois):
            m["poi_id"] = i
        checkins = [(u, [new_id[p] for p in seq if p in new_id]) for u, seq in checkins]
        checkins = [(u, seq) for u, seq in checkins if len(seq) >= 2]
        source += f"|max_pois={max_pois}"
    return pois, checkins, source


def load_real_nyc(processed_dir=None, cold_poi_ratio=0.0):
    """加载 real_data_prepare.py 产出的规范格式真实 Foursquare-NYC 数据。

    返回 (pois, checkins, test_samples, num_pois, stats, cold_pois)：
      - pois         : list[{poi_id, category, lat, lng, text}]（按连续索引 0..N-1）
      - checkins     : list[(user_id, [poi_idx,...])]  训练轨迹
      - test_samples : list[(0, [poi_idx,...history], poi_idx_target)]  （uid=0 占位，评估不使用）
      - num_pois     : 连续索引后的 POI 总数
      - stats        : 规模统计字典
      - cold_pois    : 当 cold_poi_ratio>0 时，为「训练阶段完全不可见、仅在测试做目标」的
                       冷启动 POI 索引集合；否则为空集。
    PoiId（真实值，不连续，最大≈9690）被重映射为 0..N-1 连续索引，
    以匹配 KG builder / STKGNet 的位置索引假设。

    严格 POI 冷启动设定（cold_poi_ratio>0）：
      选取训练交互频次最低、且在测试集中作为目标出现的 POI 作为冷启 POI，
      将其所有训练交互剔除，使 CF（BPR-MF/LightGCN）无法为这些 POI 学习 embedding，
      而 ours 仍可经由 KG 的类别/地理/语义边为冷启 POI 提供表征——这正是本方法
      （C1 LLM 增强 KG / C3 可解释推理）相对纯协同过滤的核心优势战场。
    """
    import os, json
    if processed_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        processed_dir = os.path.normpath(os.path.join(
            here, "..", "..", "..", "..", "data", "real_foursquare_nyc", "processed"))
    with open(os.path.join(processed_dir, "poi_meta.json"), encoding="utf-8") as f:
        poi_meta_raw = json.load(f)
    with open(os.path.join(processed_dir, "train_trajs.json"), encoding="utf-8") as f:
        train_trajs = json.load(f)
    with open(os.path.join(processed_dir, "test_pairs.json"), encoding="utf-8") as f:
        test_pairs = json.load(f)
    try:
        with open(os.path.join(processed_dir, "stats.json"), encoding="utf-8") as f:
            stats = json.load(f)
    except FileNotFoundError:
        stats = {}

    # ---- 收集所有出现的真实 PoiId，重映射为连续索引 ----
    all_ids = set(int(k) for k in poi_meta_raw.keys())
    for p in test_pairs:
        all_ids.add(int(p["target"]))
        all_ids.update(int(x) for x in p["history"])
    remap = {pid: idx for idx, pid in enumerate(sorted(all_ids))}
    num_pois = len(remap)

    # ---- 构建 pois 列表（按连续索引）----
    pois = []
    for idx in range(num_pois):
        raw = poi_meta_raw.get(str(idx), None)
        if raw is None:
            # 测试集特有 POI：meta 未知，置默认
            raw = {"lat": 0.0, "lng": 0.0, "cat_id": -1, "cat_name": "unknown"}
        cat_name = str(raw.get("cat_name", "unknown"))
        lat = float(raw.get("lat", 0.0))
        lng = float(raw.get("lng", 0.0))
        # C1 文本表示：类别名 + 粗粒度地理网格（经纬度 2 位小数≈1.1km 网格）。
        # 仅用类别名会让同类 POI 得到完全相同嵌入、语义边退化为同类团（全连），
        # 故注入位置 token 使语义表征同时编码「类型 + 邻里」，语义边更具判别力。
        text = f"{cat_name} near {lat:.2f},{lng:.2f}"
        pois.append({
            "poi_id": idx,
            "category": int(raw.get("cat_id", -1)),
            "lat": lat,
            "lng": lng,
            "text": text,
        })

    # ---- 用户 id 重映射为连续索引（真实数据 user_id 不连续，否则越界）----
    all_uids = sorted({int(tr["user_id"]) for tr in train_trajs})
    uid_remap = {u: i for i, u in enumerate(all_uids)}
    num_users_real = len(uid_remap)

    # ---- 训练 checkins（按 session 分组，已按时间序）----
    checkins = []
    for tr in train_trajs:
        seq = [remap[int(p)] for p in tr["pois"]]
        if len(seq) >= 2:
            checkins.append((uid_remap[int(tr["user_id"])], seq))
    stats["num_users_remapped"] = num_users_real

    # ---- 测试样本 ----
    test_samples = []
    for p in test_pairs:
        hist = [remap[int(x)] for x in p["history"]]
        tgt = remap[int(p["target"])]
        if hist:
            test_samples.append((0, hist, tgt))

    stats["num_pois_remapped"] = num_pois
    cold_pois = set()
    if cold_poi_ratio and cold_poi_ratio > 0:
        from collections import Counter
        train_freq = Counter(p for _, seq in checkins for p in seq)
        test_tgt_set = {t for _, _, t in test_samples}
        # 选训练交互频次最低的 POI 作为冷启候选（频次越低越接近冷启动），
        # 且仅保留在测试集中作为目标出现的 POI，保证冷启子集非空。
        sorted_by_freq = sorted(range(num_pois), key=lambda p: train_freq.get(p, 0))
        n_cold = max(1, int(round(cold_poi_ratio * num_pois)))
        cand = [p for p in sorted_by_freq[:n_cold] if p in test_tgt_set]
        cold_pois = set(cand)
        # 从训练 checkins 中剔除冷启 POI 的所有出现，使训练阶段对该 POI 完全不可见
        new_checkins = []
        for u, seq in checkins:
            s = [p for p in seq if p not in cold_pois]
            if len(s) >= 2:
                new_checkins.append((u, s))
        checkins = new_checkins
        stats["n_cold_pois"] = len(cold_pois)
        print(f"[cold-POI] 选 {len(cold_pois)} 个冷启 POI（训练频次最低且在测试出现为目标），"
              f"已从训练剔除其交互")
    return pois, checkins, test_samples, num_pois, stats, cold_pois

