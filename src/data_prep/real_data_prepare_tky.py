"""真实 Foursquare-TKY (TSMC2014 原始签到) → 规范格式转换（纯 Python，无需 torch）。

输入（github.com/ruslansco/Foursquare-Data-Analysis 镜像的 TSMC2014 原始 CSV）：
  - dataset_TSMC2014_TKY.csv : Yang et al. 2014/2015 标准 Foursquare 签到，
    列（顺序可能略有差异，本脚本按列名模糊匹配自适应）：
      User ID, Venue ID, Venue Category ID, Venue Category Name,
      Latitude, Longitude, Timezone Offset in Minutes, UTC Time

输出（data/real_foursquare_tky/processed/），与 real_foursquare_nyc/processed/ 同构：
  - poi_meta.json        : {poi_id: {lat,lng,cat_id,cat_name}}
  - train_trajs.json     : [{session_id,user_id,pois:[...时间序...],times:[...]}]
  - test_pairs.json      : [{history:[poi_id...], target:poi_id}]
  - stats.json           : 规模统计（供论文/专利引用）

协议说明（与 NYC 同构但独立构造，已在 cross_dataset_table 注明「跨数据集不比绝对值」）：
  - 每个用户按 UTC 时间排序后，以 24h 时间间隔切分为多条轨迹（session），
    模拟 next-POI 的会话级评估，与 LLM4POI 的 pseudo_session_trajectory_id 思路一致。
  - 测试集 = 每个用户的「最后一条 session」做 leave-one-out：history = 该 session
    倒数第二个及之前的 POI，target = 该 session 最后一个 POI；仅保留 history 非空者。
  - 训练集 = 全部签到，但剔除每个测试 session 的最后一个 POI（target），避免泄漏。
  - 因 HF 镜像（w11wo/LLM4POI）在本沙箱不可达，TKY 无法取到与 NYC 完全一致的官方
    1447 划分；采用上述 leave-one-out 是 POI 推荐标准协议，ours 与全部基线在同一
    TKY 测试集上公平对比即可。
"""
import json
import os
import csv
import sys
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # .../旅游推荐论文
WORKSPACE = os.path.dirname(ROOT)                  # .../2026年7月
RAW_DIR = os.path.join(WORKSPACE, "data", "real_foursquare_tky")
OUT_DIR = os.path.join(RAW_DIR, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

SESSION_GAP_SECONDS = 24 * 3600  # 轨迹切分的时间间隔阈值

# TKY 原始含 61,858 个 venue，O(N^2) 的 KG 构建（geo/类目/语义矩阵各约 30GB）会 OOM。
# 按签到频次取高频子集，与 NYC（4,980 POI）规模相当、跨域可比；论文须如实披露子采样。
# 可用环境变量 TKY_MAX_POIS 覆盖（0=不子采样，仅当机器内存足够时）。
import os as _os
MAX_POIS = int(_os.environ.get("TKY_MAX_POIS", "6000"))


def _parse_epoch(utc_str):
    """'2012-04-03T17:00:04Z' -> epoch 秒（UTC）。失败返回 0。"""
    s = (utc_str or "").strip().replace("Z", "").strip()
    if not s:
        return 0.0
    fmts = ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M", "%Y-%m-%d")
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return 0.0


def load_raw(raw_dir, max_pois=0):
    path = os.path.join(raw_dir, "dataset_TSMC2014_TKY.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 TKY 原始文件: {path}")
    poi_meta = {}
    # user -> list[(epoch, venue_id, lat, lng, cat_id, cat_name)]
    by_user = defaultdict(list)
    n_rows = 0
    # TSMC2014 的 venueId / venueCategoryId 都是 Foursquare 十六进制长串
    # （如 4c9ef97f542b224b3995f99f / 4bf58dd8d48988d1e0931735），无法直接 int()。
    # 此处将二者分别映射为连续整数：venue 映射保证「同一 POI → 同一 int」，
    # category 映射保证「同类目 → 同一 int」用于 KG 的 category 边；文本 cat_name 仍保留。
    cat_map = {}
    venue_map = {}
    next_cat = 0
    next_venue = 0

    def _map_cat(cid_str):
        nonlocal next_cat
        if cid_str not in cat_map:
            cat_map[cid_str] = next_cat
            next_cat += 1
        return cat_map[cid_str]

    def _map_venue(vid_str):
        nonlocal next_venue
        if vid_str not in venue_map:
            venue_map[vid_str] = next_venue
            next_venue += 1
        return venue_map[vid_str]

    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader)
        low = [h.lower() for h in header]
        # 精确/camelCase 自适应匹配：先精确小写匹配，再退化为子串匹配。
        # TSMC2014 标准列：userId, venueId, venueCategoryId, venueCategory,
        # latitude, longitude, timezoneOffset, utcTimestamp
        def _find(*cands):
            for c in cands:
                if c in low:
                    return header[low.index(c)]
            for c in cands:
                for i, name in enumerate(low):
                    if c in name:
                        return header[i]
            return None

        c_user = _find("userid")
        c_venue = _find("venueid")
        c_cat_id = _find("categoryid")
        c_cat_name = _find("venuecategory")
        c_lat = _find("latitude")
        c_lng = _find("longitude")
        c_utc = _find("utctimestamp")
        print(f"[prepare-tky] 列映射: user={c_user} venue={c_venue} cat_id={c_cat_id} "
              f"cat_name={c_cat_name} lat={c_lat} lng={c_lng} utc={c_utc}", flush=True)
        idxs = [header.index(x) for x in
                [c_user, c_venue, c_cat_id, c_cat_name, c_lat, c_lng, c_utc]
                if x is not None]
        for row in reader:
            if len(row) <= max(idxs):
                continue
            try:
                uid = int(float(row[header.index(c_user)]))
                vid = _map_venue(row[header.index(c_venue)].strip())
                cat_id = _map_cat(row[header.index(c_cat_id)].strip())
                lat = float(row[header.index(c_lat)])
                lng = float(row[header.index(c_lng)])
                cat_name = row[header.index(c_cat_name)].strip()
                epoch = _parse_epoch(row[header.index(c_utc)])
            except (ValueError, IndexError):
                continue
            by_user[uid].append((epoch, vid, lat, lng, cat_id, cat_name))
            if vid not in poi_meta:
                poi_meta[vid] = {"lat": lat, "lng": lng,
                                 "cat_id": cat_id, "cat_name": cat_name}
            n_rows += 1
    print(f"[prepare-tky] 读取 {n_rows} 条签到，{len(by_user)} 用户，{len(poi_meta)} POI，"
          f"{next_cat} 个类目", flush=True)

    # ---- 可选：按频次子采样高频 POI（避免 O(N^2) KG 构建 OOM，与 NYC 规模可比）----
    if max_pois and len(poi_meta) > max_pois:
        from collections import Counter
        freq = Counter(v for seq in by_user.values() for (_, v, *_r) in seq)
        keep = [v for v, _ in freq.most_common(max_pois)]
        new_vid = {v: i for i, v in enumerate(keep)}
        new_poi_meta = {new_vid[v]: poi_meta[v] for v in keep}
        new_by_user = {}
        for u, seq in by_user.items():
            ns = [(ep, new_vid[v], lat, lng, cat, cname)
                  for (ep, v, lat, lng, cat, cname) in seq if v in new_vid]
            if len(ns) >= 2:
                new_by_user[u] = ns
        poi_meta, by_user = new_poi_meta, new_by_user
        print(f"[prepare-tky] 子采样 POI → 保留高频 {len(poi_meta)} 个，剩 {len(by_user)} 用户",
              flush=True)
    return poi_meta, by_user


def split_sessions(seq):
    """seq: list[(epoch, vid, ...)] 已按时间排序 → 按 24h 间隔切成多条 session。"""
    sessions = []
    cur = []
    last_t = None
    for item in seq:
        t = item[0]
        if last_t is not None and (t - last_t) > SESSION_GAP_SECONDS:
            if cur:
                sessions.append(cur)
            cur = []
        cur.append(item)
        last_t = t
    if cur:
        sessions.append(cur)
    return sessions


def main():
    print(f"[prepare-tky] 读取原始数据: {RAW_DIR}", flush=True)
    poi_meta, by_user = load_raw(RAW_DIR, MAX_POIS)

    train_trajs = []
    test_pairs = []
    sid = 0
    n_train_checkins = 0
    for uid, seq in by_user.items():
        seq.sort(key=lambda x: x[0])  # 按 UTC 时间排序
        sessions = split_sessions(seq)
        if not sessions:
            continue
        # 测试 session = 最后一条；其最后一个 POI 作 target
        test_sess = sessions[-1]
        train_sessions = sessions[:-1]
        # 若用户只有一条 session，仍取该 session 的最后一个 POI 作 target，
        # 但 history 需非空（session 长度 ≥ 2）
        if not train_sessions:
            train_sessions = []
            # target 仍来自 test_sess，history = 之前的 POI
        # 构造测试对（leave-one-out）
        if len(test_sess) >= 2:
            hist = [it[1] for it in test_sess[:-1]]
            tgt = test_sess[-1][1]
            test_pairs.append({"history": hist, "target": tgt})
        # 训练轨迹：所有 session，但剔除测试 session 的最后一个 POI（target）
        for s in train_sessions:
            pois = [it[1] for it in s]
            times = [it[0] for it in s]
            if len(pois) >= 2:
                train_trajs.append({"session_id": sid, "user_id": uid,
                                    "pois": pois, "times": times})
                n_train_checkins += len(pois)
                sid += 1
        # 若测试 session 也是唯一 session，其 history 部分（除 target）仍进训练
        if not train_sessions and len(test_sess) >= 2:
            pois = [it[1] for it in test_sess[:-1]]
            times = [it[0] for it in test_sess[:-1]]
            if len(pois) >= 2:
                train_trajs.append({"session_id": sid, "user_id": uid,
                                    "pois": pois, "times": times})
                n_train_checkins += len(pois)
                sid += 1

    # 测试 target / history 中出现的 POI 可能超出 poi_meta（应并入）
    all_poi = set(poi_meta.keys())
    for p in test_pairs:
        all_poi.add(p["target"])
        all_poi.update(p["history"])
    for pid in all_poi:
        if pid not in poi_meta:
            poi_meta[pid] = {"lat": 0.0, "lng": 0.0, "cat_id": -1, "cat_name": "unknown"}

    lens = [len(t["pois"]) for t in train_trajs]
    stats = {
        "n_pois": len(poi_meta),
        "n_users": len(by_user),
        "n_train_trajs": len(train_trajs),
        "n_train_checkins": n_train_checkins,
        "avg_traj_len": round(sum(lens) / max(1, len(lens)), 2),
        "max_traj_len": max(lens) if lens else 0,
        "n_test_pairs": len(test_pairs),
        "n_categories": len({v["cat_id"] for v in poi_meta.values() if v["cat_id"] >= 0}),
        "session_gap_seconds": SESSION_GAP_SECONDS,
        "protocol": "leave-one-out last session per user; 24h session split",
    }
    with open(os.path.join(OUT_DIR, "poi_meta.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in poi_meta.items()}, f, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "train_trajs.json"), "w", encoding="utf-8") as f:
        json.dump(train_trajs, f)
    with open(os.path.join(OUT_DIR, "test_pairs.json"), "w", encoding="utf-8") as f:
        json.dump(test_pairs, f)
    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("[prepare-tky] 完成。统计:", flush=True)
    for k, v in stats.items():
        print(f"  {k}: {v}", flush=True)
    print(f"[prepare-tky] 输出目录: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
