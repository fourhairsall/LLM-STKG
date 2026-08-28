"""通用序列推荐数据集加载器（SASRec 基准：MovieLens-1M / Gowalla / Steam）。

目标：把三类「非 POI 或异源 POI」数据集统一转换为与
``foursquare_loader.load_real_nyc`` 同构的返回：
    (pois, checkins, test_samples, num_pois, stats, cold_pois)

其中：
  - pois         : list[{poi_id, category, lat, lng, text}]
                  text 字段将喂给 BGE 做语义底物（Steam 置空 → 触发 ID 嵌入兜底）。
  - checkins    : list[(user_id, [poi_idx,...])]  训练轨迹（每个用户序列去掉 test 目标）
  - test_samples: list[(uid, [history_poi_idx], target_poi_idx)]  leave-one-out 全候选评测
  - cold_pois   : 本实现默认空集（跨域冷启动留作后续扩展）

关键适配（用户决策 2026-08-02）：
  - MovieLens-1M：text = "Title Genres"（真实 BGE 语义）；无地理 → C6 邻近改为共现。
  - Gowalla     ：text = "poi near lat,lng"（地理语义，同 Foursquare 方案）；保留距离邻近。
  - Steam       ：text = ""（无文本/类目/地理）→ 语义退回可学习 ID 嵌入（w/o LLM-text 消融）。

所有数据集均按频次子采样到可训练规模（max_pois / max_users），并在 stats 中如实记录
原始规模与子采样规模，便于论文诚实标注。

共现矩阵 cooc_matrix.npy [N,N] 一并保存到 processed 目录，供 C6 的 "cooc" 通道使用。
"""
import os
import json
import math
from collections import defaultdict, Counter

# 默认子采样规模（与 Foursquare-NYC 实验量级对齐，保证 GPU 可训练）
DEFAULT_MAX_POIS = 5000
DEFAULT_MAX_USERS = 20000
DEFAULT_SEQ_CAP = 200          # history 截断长度（对齐 Foursquare 协议 seq_len=200）


# --------------------------------------------------------------------------
# 公共工具
# --------------------------------------------------------------------------
def _remap_and_build_pois(raw_pois, raw_checkins, max_pois=None, max_users=None):
    """对 (raw_pois: dict[real_id]->meta, raw_checkins: list[(uid, seq)]) 做
    频次子采样 + 连续索引重映射，返回 (pois, checkins, num_pois, stats, id_map)。

    raw_pois[real_id] = {"category","lat","lng","text"}
    """
    # 1) POI 频次子采样（保留高频 POI，控制图规模）
    if max_pois:
        cnt = Counter(p for _, seq in raw_checkins for p in seq)
        keep = set(i for i, _ in cnt.most_common(max_pois))
    else:
        keep = set(raw_pois.keys())
    # 按真实 id 排序后连续重映射，保证可复现
    keep_sorted = sorted(keep)
    id_map = {rid: idx for idx, rid in enumerate(keep_sorted)}
    num_pois = len(id_map)

    pois = []
    for idx in range(num_pois):
        rid = keep_sorted[idx]
        m = raw_pois.get(rid, {"category": -1, "lat": 0.0, "lng": 0.0, "text": ""})
        pois.append({
            "poi_id": idx,
            "category": int(m.get("category", -1)),
            "lat": float(m.get("lat", 0.0)),
            "lng": float(m.get("lng", 0.0)),
            "text": str(m.get("text", "")),
        })

    # 2) 用户子采样（按活跃度）
    if max_users:
        user_cnt = Counter(u for u, seq in raw_checkins for _ in seq)
        keep_u = set(u for u, _ in user_cnt.most_common(max_users))
        raw_checkins = [(u, seq) for u, seq in raw_checkins if u in keep_u]

    # 3) 连续用户索引
    all_uids = sorted({u for u, seq in raw_checkins})
    uid_map = {u: i for i, u in enumerate(all_uids)}
    num_users = len(uid_map)

    checkins = []
    for u, seq in raw_checkins:
        s = [id_map[p] for p in seq if p in id_map]
        if len(s) >= 2:
            checkins.append((uid_map[u], s))

    stats = {
        "num_pois": num_pois,
        "num_users": num_users,
        "num_checkins": len(checkins),
        "num_interactions": sum(len(s) for _, s in checkins),
    }
    return pois, checkins, num_pois, num_users, stats, id_map


def _leave_one_out_split(checkins, seq_cap=DEFAULT_SEQ_CAP):
    """对每个用户序列做 leave-one-out：
      train_checkin = seq[:-1]（模型在训练时以 seq[:-2] 预测 seq[:-1]）
      val_sample    = (uid, seq[:-2][-cap:], seq[-2])   （L>=4 才有）
      test_sample   = (uid, seq[:-1][-cap:], seq[-1])   （L>=3 才有）
    返回 (train_checkins, val_samples, test_samples)。
    """
    train_checkins, val_samples, test_samples = [], [], []
    for u, seq in checkins:
        L = len(seq)
        if L >= 3:
            train_checkins.append((u, seq[:-1]))
            test_samples.append((u, seq[:-1][-seq_cap:], seq[-1]))
        if L >= 4:
            val_samples.append((u, seq[:-2][-seq_cap:], seq[-2]))
    return train_checkins, val_samples, test_samples


def _build_cooc_matrix(checkins, num_pois, smoothing=1.0):
    """会话内共现强度矩阵 [N,N]。
    共现定义：同一用户序列中两两 POI 同时出现次数（无序对，去重每次会话计数 1）。
    归一化：cooc[c,h] = log(1 + cooc_raw[c,h]) / log(1 + max_cooc_in_row)  → 行归一化到 [0,1]。
    """
    pair = Counter()
    for _, seq in checkins:
        seen = set(seq)
        # 去重后两两组合
        uniq = list(seen)
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = uniq[i], uniq[j]
                pair[(a, b)] += 1
                pair[(b, a)] += 1
    import numpy as np
    M = np.zeros((num_pois, num_pois), dtype=np.float32)
    for (a, b), c in pair.items():
        if a < num_pois and b < num_pois:
            M[a, b] = math.log1p(c)
    # 行归一化（按每行最大值），避免热门 POI 共现值压倒
    rowmax = M.max(axis=1, keepdims=True)
    rowmax[rowmax == 0] = 1.0
    M = M / rowmax
    return M


def _refilter_by_train_freq(checkins_full, train_c, val_s, test_s, pois, min_freq, name):
    """基于 *训练* 频次过滤低频 POI（5-core 风格核心子集）。
    keep = POI whose frequency in train_c >= min_freq. 低频 POI 从候选与 target 中彻底
    移除并重新连续索引；所有序列/样本/cooc 同步重映射。返回与 _remap_and_build_pois
    后处理一致的结构 (pois, checkins_full, train_c, val_s, test_s, num_pois, cooc, stats)。
    """
    cnt = Counter(p for _, seq in train_c for p in seq)
    keep = {p for p, c in cnt.items() if c >= min_freq}
    keep_sorted = sorted(keep)
    id_map = {rid: idx for idx, rid in enumerate(keep_sorted)}
    num_pois = len(id_map)
    pois_new = [pois[rid] for rid in keep_sorted]

    def remap_seq(seq):
        return [id_map[p] for p in seq if p in id_map]

    checkins_full_new = [(u, remap_seq(s)) for u, s in checkins_full if len(remap_seq(s)) >= 2]
    train_c_new = [(u, remap_seq(s)) for u, s in train_c if len(remap_seq(s)) >= 2]

    def filt(samples):
        out = []
        for u, h, t in samples:
            if t not in id_map:
                continue
            hh = remap_seq(h)
            out.append((u, hh, id_map[t]))
        return out

    val_s_new = filt(val_s)
    test_s_new = filt(test_s)
    cooc = _build_cooc_matrix(train_c_new, num_pois)
    # 重新连续映射 uid（仅标识，保证紧凑）
    all_uids = sorted({u for u, _ in checkins_full_new}
                     | {u for u, _, _ in val_s_new}
                     | {u for u, _, _ in test_s_new})
    uid_map = {u: i for i, u in enumerate(all_uids)}
    checkins_full_new = [(uid_map[u], s) for u, s in checkins_full_new]
    train_c_new = [(uid_map[u], s) for u, s in train_c_new]
    val_s_new = [(uid_map[u], h, t) for u, h, t in val_s_new]
    test_s_new = [(uid_map[u], h, t) for u, h, t in test_s_new]
    stats = {
        "num_pois": num_pois,
        "num_users": len(uid_map),
        "num_checkins": len(train_c_new),
        "num_interactions": sum(len(s) for _, s in train_c_new),
    }
    print(f"[{name}] train_freq>={min_freq} -> POIs={num_pois} train_seq={len(train_c_new)} "
          f"test={len(test_s_new)} val={len(val_s_new)}", flush=True)
    return (pois_new, checkins_full_new, train_c_new, val_s_new, test_s_new,
            num_pois, cooc, stats)


def _save_processed(out_dir, pois, train_checkins, val_samples, test_samples,
                    stats, cooc, name):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "pois.json"), "w", encoding="utf-8") as f:
        json.dump(pois, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "train_checkins.json"), "w", encoding="utf-8") as f:
        json.dump([{"user_id": u, "pois": s} for u, s in train_checkins], f)
    with open(os.path.join(out_dir, "val_samples.json"), "w", encoding="utf-8") as f:
        json.dump([{"user_id": u, "history": h, "target": t} for u, h, t in val_samples], f)
    with open(os.path.join(out_dir, "test_samples.json"), "w", encoding="utf-8") as f:
        json.dump([{"user_id": u, "history": h, "target": t} for u, h, t in test_samples], f)
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    import numpy as np
    np.save(os.path.join(out_dir, "cooc_matrix.npy"), cooc)
    print(f"[{name}] 写出 processed -> {out_dir}")
    print(f"    pois={stats['num_pois']} users={stats['num_users']} "
          f"train_seq={stats['num_checkins']} test={len(test_samples)} val={len(val_samples)}")


# --------------------------------------------------------------------------
# MovieLens-1M
# --------------------------------------------------------------------------
def load_movielens(ml_dir, max_pois=DEFAULT_MAX_POIS, max_users=DEFAULT_MAX_USERS,
                   seq_cap=DEFAULT_SEQ_CAP, out_dir=None, name="movielens-1m"):
    """ml_dir 含 ratings.dat (UserID::MovieID::Rating::Timestamp) 与
    movies.dat (MovieID::Title::Genres|...)。序列按 Timestamp 升序。
    """
    # 读 movies
    movie_meta = {}
    genre_vocab = {}   # 有界类目词表：首类目 → 紧凑 id（避免 hash%1e5 导致 num_cats 爆炸 + 巨型类目团）
    def _gid(g):
        if g not in genre_vocab:
            genre_vocab[g] = len(genre_vocab)
        return genre_vocab[g]
    with open(os.path.join(ml_dir, "movies.dat"), encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) < 3:
                continue
            mid, title, genres = parts[0], parts[1], parts[2]
            first_genre = genres.split("|")[0]
            movie_meta[int(mid)] = {
                "category": _gid(first_genre),                          # 有界类目 id（≤18）
                "lat": 0.0, "lng": 0.0,
                "text": f"{title} {genres.replace('|', ' ')}",         # 真实 BGE 语义
            }
    # 读 ratings → 按用户分组并按时间排序
    raw = defaultdict(list)
    with open(os.path.join(ml_dir, "ratings.dat"), encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) < 4:
                continue
            uid, mid, _, ts = int(parts[0]), int(parts[1]), parts[2], int(parts[3])
            raw[uid].append((ts, mid))
    raw_checkins = []
    for uid, lst in raw.items():
        lst.sort()
        seq = [m for _, m in lst]
        raw_checkins.append((uid, seq))

    pois, checkins, num_pois, num_users, stats, _ = _remap_and_build_pois(
        movie_meta, raw_checkins, max_pois=max_pois, max_users=max_users)
    stats["dataset"] = name
    stats["source_users"] = len(raw)
    stats["source_movies"] = len(movie_meta)
    train_c, val_s, test_s = _leave_one_out_split(checkins, seq_cap=seq_cap)
    cooc = _build_cooc_matrix(train_c, num_pois)
    if out_dir:
        _save_processed(out_dir, pois, train_c, val_s, test_s, stats, cooc, name)
    # 转成与 load_real_nyc 同构的返回（cold_pois 空集）
    test_samples = [(0, h, t) for _, h, t in test_s]   # uid 占位 0（评测不使用）
    return pois, train_c, test_samples, num_pois, stats, set(), cooc


# --------------------------------------------------------------------------
# Gowalla（SNAP loc-gowalla_totalCheckins.txt.gz）
# --------------------------------------------------------------------------
def load_gowalla(gz_path, max_pois=DEFAULT_MAX_POIS, max_users=DEFAULT_MAX_USERS,
                 seq_cap=DEFAULT_SEQ_CAP, out_dir=None, name="gowalla"):
    """gz 文件每行：user \\t timestamp(ISO) \\t lat \\t lng \\t location_id。
    序列按时间升序。POI text = "poi near lat,lng"（地理语义，同 Foursquare）。
    """
    import gzip
    raw_pois = {}
    raw = defaultdict(list)
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            try:
                uid = int(parts[0])
                # parts[1] 为 ISO 时间戳，排序用原始字符串（字典序≈时间序对 ISO 成立）
                ts = parts[1]
                lat = float(parts[2])
                lng = float(parts[3])
                loc = int(parts[4])
            except (ValueError, IndexError):
                continue
            raw[uid].append((ts, loc, lat, lng))
            if loc not in raw_pois:
                raw_pois[loc] = {
                    "category": 0,   # 无类目：统一=0（跨域时一并关闭类目边，避免 -1 非法索引）
                    "lat": lat, "lng": lng,
                    "text": f"poi near {lat:.2f},{lng:.2f}",   # 地理语义
                }
    raw_checkins = []
    for uid, lst in raw.items():
        lst.sort(key=lambda x: x[0])
        seq = [loc for _, loc, _, _ in lst]
        raw_checkins.append((uid, seq))

    pois, checkins, num_pois, num_users, stats, _ = _remap_and_build_pois(
        raw_pois, raw_checkins, max_pois=max_pois, max_users=max_users)
    stats["dataset"] = name
    stats["source_users"] = len(raw)
    stats["source_locations"] = len(raw_pois)
    train_c, val_s, test_s = _leave_one_out_split(checkins, seq_cap=seq_cap)
    cooc = _build_cooc_matrix(train_c, num_pois)
    if out_dir:
        _save_processed(out_dir, pois, train_c, val_s, test_s, stats, cooc, name)
    test_samples = [(0, h, t) for _, h, t in test_s]
    return pois, train_c, test_samples, num_pois, stats, set(), cooc


# --------------------------------------------------------------------------
# Steam（SASRec data/Steam.txt：每行 "user item"，全局按时间排序）
# --------------------------------------------------------------------------
def load_steam(txt_path, max_pois=DEFAULT_MAX_POIS, max_users=DEFAULT_MAX_USERS,
               seq_cap=DEFAULT_SEQ_CAP, out_dir=None, name="steam"):
    """每行 "user item"（按时间全局有序）。同一 user 的连续行即其交互序列。
    无文本/类目/地理 → text="" 触发 ID 嵌入兜底（w/o LLM-text 消融）。
    """
    raw_pois = {}
    raw = defaultdict(list)
    cur_uid = None
    cur_seq = []
    def _flush(u, s):
        if u is not None and len(s) >= 2:
            raw[u].append(s)
    with open(txt_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                uid = int(parts[0]); item = int(parts[1])
            except ValueError:
                continue
            if uid != cur_uid:
                _flush(cur_uid, cur_seq)
                cur_uid = uid; cur_seq = []
            cur_seq.append(item)
            if item not in raw_pois:
                # 无类目：统一类目=0（避免 -1 触发 Embedding/索引非法）；跨域时一并关闭类目边。
                raw_pois[item] = {"category": 0, "lat": 0.0, "lng": 0.0, "text": ""}
        _flush(cur_uid, cur_seq)

    # 子采样：Steam 规模巨大，先按 POI/用户频次裁剪再建序列
    raw_checkins = [(u, seq) for u, seqs in raw.items() for seq in seqs]

    pois, checkins, num_pois, num_users, stats, _ = _remap_and_build_pois(
        raw_pois, raw_checkins, max_pois=max_pois, max_users=max_users)
    stats["dataset"] = name
    stats["source_users"] = len(raw)
    stats["source_games"] = len(raw_pois)
    train_c, val_s, test_s = _leave_one_out_split(checkins, seq_cap=seq_cap)
    cooc = _build_cooc_matrix(train_c, num_pois)
    if out_dir:
        _save_processed(out_dir, pois, train_c, val_s, test_s, stats, cooc, name)
    test_samples = [(0, h, t) for _, h, t in test_s]
    return pois, train_c, test_samples, num_pois, stats, set(), cooc


# --------------------------------------------------------------------------
# Steam-200k（Kaggle tamber/steam-video-games，含 game-title 真实文本）
# --------------------------------------------------------------------------
def load_steam200k(csv_path, max_pois=DEFAULT_MAX_POIS, max_users=DEFAULT_MAX_USERS,
                   seq_cap=DEFAULT_SEQ_CAP, out_dir=None, name="steam200k"):
    """Kaggle Steam-200k：列 user-id, game-title, behavior-name, value[, extra]。
    无时间戳 -> 同用户内行序即交互序。POI text = game-title（真实文本，可灌 BGE），
    与无文本的 SASRec-Steam(600) 形成「同族文本对照」：LLM4POI-style 在无文本域塌缩、
    在文本域(Steam-200k)生效，直接证成文本必要性。

    序列构造：同用户相邻相同游戏去重（purchase/play 两行 -> 一条），保留交互顺序。
    """
    import csv
    raw_pois = {}
    raw = defaultdict(list)
    cur_uid = None
    cur_seq = []

    def _flush(u, s):
        if u is not None and len(s) >= 2:
            raw[u].append(s)

    with open(csv_path, encoding="utf-8", errors="ignore", newline="") as f:
        for parts in csv.reader(f):
            if len(parts) < 4:
                continue
            try:
                uid = int(parts[0])
                title = parts[1].strip()
            except (ValueError, IndexError):
                continue
            if not title:
                continue
            if uid != cur_uid:
                _flush(cur_uid, cur_seq)
                cur_uid = uid
                cur_seq = []
            if not cur_seq or cur_seq[-1] != title:
                cur_seq.append(title)
            if title not in raw_pois:
                # 真实文本标签（BGE 语义底物）；跨域 ours 仍走 w/o LLM-text 模式（统一对照）
                raw_pois[title] = {"category": 0, "lat": 0.0, "lng": 0.0, "text": title}
        _flush(cur_uid, cur_seq)

    raw_checkins = [(u, seq) for u, seqs in raw.items() for seq in seqs]
    pois, checkins, num_pois, num_users, stats, _ = _remap_and_build_pois(
        raw_pois, raw_checkins, max_pois=max_pois, max_users=max_users)
    stats["dataset"] = name
    stats["source_users"] = len(raw)
    stats["source_games"] = len(raw_pois)
    train_c, val_s, test_s = _leave_one_out_split(checkins, seq_cap=seq_cap)
    cooc = _build_cooc_matrix(train_c, num_pois)
    if out_dir:
        _save_processed(out_dir, pois, train_c, val_s, test_s, stats, cooc, name)
    test_samples = [(0, h, t) for _, h, t in test_s]
    return pois, train_c, test_samples, num_pois, stats, set(), cooc


# --------------------------------------------------------------------------
# Amazon Beauty（Amazon Reviews 2023, McAuley-Lab "All_Beauty" 类目，含富文本）
# --------------------------------------------------------------------------
def load_amazon_beauty(meta_parquet, csv_list, max_pois=DEFAULT_MAX_POIS,
                       max_users=DEFAULT_MAX_USERS, seq_cap=DEFAULT_SEQ_CAP,
                       min_freq=1, out_dir=None, name="amazon_beauty"):
    """Amazon Reviews 2023 的 All_Beauty 类目（即「Amazon Beauty」）。
    meta_parquet: 商品元数据 (title/description/categories/main_category/parent_asin)
    csv_list: [train.csv, valid.csv, test.csv] (user_id, parent_asin, rating, timestamp)
    同用户跨 split 合并、按 timestamp 升序成序列。POI text = title + main_category + desc[:150]
    （富文本，可灌 BGE；LLM4POI-style 在此域可正常生效，用作文本域对照）。

    ours 仍统一走 w/o LLM-text 模式（行为+结构），与跨域其它域一致；文本仅供给
    LLM4POI-style 基线做语义种子，构成「文本域全候选头对头」的公平对照。
    """
    import pandas as pd
    import numpy as np
    # 1) 元数据 -> POI 文本
    meta = pd.read_parquet(meta_parquet)
    raw_pois = {}
    for _, row in meta.iterrows():
        asin = str(row.get("parent_asin", "") or "").strip()
        if not asin:
            continue
        title = str(row.get("title", "") or "")
        cat = str(row.get("main_category", "") or "")
        desc_val = row.get("description", "")
        if desc_val is None or (isinstance(desc_val, (list, tuple, np.ndarray)) and len(desc_val) == 0):
            desc = ""
        elif isinstance(desc_val, (list, tuple, np.ndarray)):
            desc = " ".join(str(x) for x in desc_val)
        else:
            desc = str(desc_val)
        text = (str(title) + ". Category: " + cat + ". " + desc[:150]).strip()
        raw_pois[asin] = {"category": 0, "lat": 0.0, "lng": 0.0, "text": text}
    # 2) 交互（合并三 split，按时间排序）
    raw = defaultdict(list)
    for csv_path in csv_list:
        df = pd.read_csv(csv_path, dtype={"user_id": str, "parent_asin": str})
        for uid, asin, ts in zip(df["user_id"], df["parent_asin"], df["timestamp"]):
            raw[str(uid)].append((int(ts), str(asin)))
    # 3) 序列 + 缺失 meta 的 asin 兜底文本
    raw_checkins = []
    for uid, lst in raw.items():
        lst.sort(key=lambda x: x[0])
        seq = [a for _, a in lst]
        seq = [seq[0]] + [s for i, s in enumerate(seq[1:], 1) if s != seq[i - 1]]
        for a in seq:
            if a not in raw_pois:
                raw_pois[a] = {"category": 0, "lat": 0.0, "lng": 0.0, "text": a}
        if len(seq) >= 2:
            raw_checkins.append((uid, seq))
    pois, checkins, num_pois, num_users, stats, _ = _remap_and_build_pois(
        raw_pois, raw_checkins, max_pois=max_pois, max_users=max_users)
    stats["dataset"] = name
    stats["source_users"] = len(raw)
    stats["source_items"] = len(raw_pois)
    train_c, val_s, test_s = _leave_one_out_split(checkins, seq_cap=seq_cap)
    # 3.5) 基于 *训练* 频次过滤低频 POI（5-core 风格核心子集：确保候选 POI 在训练中充分
    #      出现，既不作候选也不作 target，改善极稀疏评分数据的 next-POI 可评性与评估公平性）
    if min_freq and min_freq > 1:
        pois, checkins, train_c, val_s, test_s, num_pois, cooc, stats = _refilter_by_train_freq(
            checkins, train_c, val_s, test_s, pois, min_freq, name)
    else:
        cooc = _build_cooc_matrix(train_c, num_pois)
    if out_dir:
        _save_processed(out_dir, pois, train_c, val_s, test_s, stats, cooc, name)
    test_samples = [(0, h, t) for _, h, t in test_s]
    return pois, train_c, test_samples, num_pois, stats, set(), cooc


# --------------------------------------------------------------------------
# CLI：一次性处理全部数据集
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"))
    ap.add_argument("--max_pois", type=int, default=DEFAULT_MAX_POIS)
    ap.add_argument("--max_users", type=int, default=DEFAULT_MAX_USERS)
    ap.add_argument("--only", default=None,
                    help="process only one dataset: movielens|gowalla|steam|steam200k|amazon_beauty "
                         "(default: all)")
    ap.add_argument("--min_freq", type=int, default=1,
                    help="(amazon_beauty) keep only POIs with train frequency >= this; "
                         "5-core style core subset to improve next-POI evaluability")
    args = ap.parse_args()
    root = args.data_root

    # MovieLens-1M
    if args.only in (None, "movielens"):
        ml_dir = os.path.join(root, "ml-1m", "ml-1m")
        if os.path.isdir(ml_dir):
            load_movielens(ml_dir, max_pois=args.max_pois, max_users=args.max_users,
                           out_dir=os.path.join(root, "ml-1m", "processed"), name="movielens-1m")
        else:
            print(f"[skip] MovieLens dir not found: {ml_dir}")

    # Gowalla
    if args.only in (None, "gowalla"):
        gw = os.path.join(root, "gowalla", "loc-gowalla_totalCheckins.txt.gz")
        if os.path.exists(gw):
            load_gowalla(gw, max_pois=args.max_pois, max_users=args.max_users,
                         out_dir=os.path.join(root, "gowalla", "processed"), name="gowalla")
        else:
            print(f"[skip] Gowalla file not found: {gw}")

    # Steam
    if args.only in (None, "steam"):
        st = os.path.join(root, "steam", "Steam.txt")
        if os.path.exists(st):
            load_steam(st, max_pois=args.max_pois, max_users=args.max_users,
                       out_dir=os.path.join(root, "steam", "processed"), name="steam")
        else:
            print(f"[skip] Steam file not found: {st}")

    # Steam-200k（含 game-title 文本）
    if args.only in (None, "steam200k"):
        s2 = os.path.join(root, "steam200k", "steam-200k.csv")
        if os.path.exists(s2):
            load_steam200k(s2, max_pois=args.max_pois, max_users=args.max_users,
                           out_dir=os.path.join(root, "steam200k", "processed"), name="steam200k")
        else:
            print(f"[skip] Steam-200k file not found: {s2}")

    # Amazon Beauty（All_Beauty, 2023, 含富文本）
    if args.only in (None, "amazon_beauty"):
        meta = os.path.join(root, "amazon_beauty", "meta_All_Beauty.parquet")
        csvs = [os.path.join(root, "amazon_beauty", f"All_Beauty.{s}.csv")
                for s in ("train", "valid", "test")]
        if os.path.exists(meta) and all(os.path.exists(c) for c in csvs):
            load_amazon_beauty(meta, csvs, max_pois=args.max_pois, max_users=args.max_users,
                               min_freq=args.min_freq,
                               out_dir=os.path.join(root, "amazon_beauty", "processed"),
                               name="amazon_beauty")
        else:
            print(f"[skip] Amazon Beauty files not found: meta={os.path.exists(meta)} "
                  f"csvs={[os.path.exists(c) for c in csvs]}")
