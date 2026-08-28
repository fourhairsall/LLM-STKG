"""真实 Foursquare-NYC (LLM4POI 预处理版) → 规范格式转换（纯 Python，无需 torch）。

输入（HF 镜像 w11wo/LLM4POI 的 nyc/preprocessed/）：
  - train_sample.csv     : 82k+ 条签到，字段见 README
  - test_qa_pairs_kqt.txt: 1447 个 next-POI 测试样本（QA 文本格式）

输出（data/real_foursquare_nyc/processed/）：
  - poi_meta.json        : {poi_id: {lat,lng,cat_id,cat_name}}
  - train_trajs.json     : [{session_id,user_id,pois:[...时间序...],times:[...]}]
  - test_pairs.json       : [{history:[poi_id...], target:poi_id}]
  - stats.json           : 规模统计（供论文/专利引用）

规范说明：
  - 轨迹按 pseudo_session_trajectory_id 分组（标准 next-POI 协议，非按 user）。
  - 测试集采用数据集官方 test 划分（1447 样本），与训练集互斥，避免泄漏。
  - 该划分即 CoMaPOI/CaST-POI/RALLM-POI 所用 Foursquare-NYC 的同源数据；
    若需与某篇 SOTA 的「精确」划分对齐，见 head_to_head.py 的 --align 开关说明。
"""
import json
import os
import re
import csv
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # .../code
WORKSPACE = os.path.dirname(os.path.dirname(ROOT)) # .../2026年7月
RAW_DIR = os.path.join(WORKSPACE, "data", "real_foursquare_nyc")
OUT_DIR = os.path.join(RAW_DIR, "processed")
os.makedirs(OUT_DIR, exist_ok=True)


def load_train(raw_dir):
    path = os.path.join(raw_dir, "train_sample.csv")
    poi_meta = {}
    sess = defaultdict(list)  # session_id -> list of (epoch, poi_id, user_id, lat,lng,cat_id,cat_name)
    users = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(row["UTCTimeOffsetEpoch"])
                poi_id = int(float(row["PoiId"]))
                uid = int(row["UserId"])
                sid = int(row["pseudo_session_trajectory_id"])
                lat = float(row["Latitude"])
                lng = float(row["Longitude"])
                cat_id = int(row["PoiCategoryId"])
                cat_name = row["PoiCategoryName"].strip()
            except (KeyError, ValueError):
                continue
            users.add(uid)
            sess[sid].append((epoch, poi_id, uid, lat, lng, cat_id, cat_name))
            if poi_id not in poi_meta:
                poi_meta[poi_id] = {"lat": lat, "lng": lng,
                                     "cat_id": cat_id, "cat_name": cat_name}
    # 排序并输出
    trajs = []
    for sid, items in sess.items():
        items.sort(key=lambda x: x[0])
        uids = {it[2] for it in items}
        trajs.append({
            "session_id": sid,
            "user_id": next(iter(uids)),
            "pois": [it[1] for it in items],
            "times": [it[0] for it in items],
        })
    return poi_meta, trajs, users


QA_RE = re.compile(r"POI id (\d+)")


def load_test(raw_dir):
    path = os.path.join(raw_dir, "test_qa_pairs_kqt.txt")
    pairs = []
    buf = ""
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "<answer>:" in line:
                # 合并缓冲（处理 QA 跨行）
                full = (buf + line) if buf else line
                q_part, a_part = full.split("<answer>:", 1)
                hist = [int(m) for m in QA_RE.findall(q_part)]
                ans = QA_RE.findall(a_part)
                if hist and ans:
                    pairs.append({"history": hist, "target": int(ans[0])})
                buf = ""
            else:
                buf = (buf + " " + line.strip()) if buf else line.strip()
    # 处理文件末尾无 <answer>: 的残留
    if buf:
        pass
    return pairs


def main():
    print(f"[prepare] 读取原始数据: {RAW_DIR}")
    poi_meta, trajs, users = load_train(RAW_DIR)
    test_pairs = load_test(RAW_DIR)

    # 统计
    poi_ids = set(poi_meta.keys())
    # 测试 target / history 中出现的 POI 可能超出 poi_meta（应并入）
    all_poi = set(poi_ids)
    for p in test_pairs:
        all_poi.add(p["target"])
        all_poi.update(p["history"])
    # 为测试集新出现的 POI 补 meta（从 QA 文本无法拿到经纬度/类目，置 0）
    for pid in all_poi:
        if pid not in poi_meta:
            poi_meta[pid] = {"lat": 0.0, "lng": 0.0, "cat_id": -1, "cat_name": "unknown"}

    lens = [len(t["pois"]) for t in trajs]
    stats = {
        "n_pois": len(poi_meta),
        "n_users": len(users),
        "n_train_trajs": len(trajs),
        "n_train_checkins": sum(lens),
        "avg_traj_len": round(sum(lens) / max(1, len(lens)), 2),
        "max_traj_len": max(lens) if lens else 0,
        "n_test_pairs": len(test_pairs),
        "n_categories": len({v["cat_id"] for v in poi_meta.values() if v["cat_id"] >= 0}),
    }
    # 保存
    with open(os.path.join(OUT_DIR, "poi_meta.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in poi_meta.items()}, f, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "train_trajs.json"), "w", encoding="utf-8") as f:
        json.dump(trajs, f)
    with open(os.path.join(OUT_DIR, "test_pairs.json"), "w", encoding="utf-8") as f:
        json.dump(test_pairs, f)
    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("[prepare] 完成。统计:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"[prepare] 输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
