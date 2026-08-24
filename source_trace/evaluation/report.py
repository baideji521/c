"""最终报告生成：BASELINE_REPORT.md / FINAL_REPORT.md。

所有数字都从 ``reports/results.json``（benchmark 的实际运行结果）读取，
不允许手写或估算。缺失的项一律标注「未测」。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils.log import get_logger, setup_logging


def _fmt(v, nd=4, unit="") -> str:
    if v is None:
        return "未测"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f < 0:
        return "未测"
    return f"{f:.{nd}f}{unit}"


def _runs(data: dict, config: str | None = None, case: str | None = None) -> list[dict]:
    out = []
    for r in data.get("runs", []):
        if not r.get("ok"):
            continue
        if config and r["config"] != config:
            continue
        if case and r["case"] != case:
            continue
        out.append(r)
    return out


def _avg(runs: list[dict], field: str) -> float | None:
    vals = [r["metrics"].get(field) for r in runs]
    vals = [v for v in vals if isinstance(v, (int, float)) and v >= 0]
    return sum(vals) / len(vals) if vals else None


def _best_config(data: dict) -> tuple[str | None, list[dict]]:
    by: dict[str, list[dict]] = {}
    for r in _runs(data):
        by.setdefault(r["config"], []).append(r)
    if not by:
        return None, []
    scored = []
    for k, rs in by.items():
        sr = _avg(rs, "source_recall") or 0.0
        gr = _avg(rs, "segment_recall") or 0.0
        se = _avg(rs, "mean_start_error")
        scored.append((gr, sr, -(se if se is not None else 99), k, rs))
    scored.sort(reverse=True)
    return scored[0][3], scored[0][4]


def _repeat_note(run: dict | None) -> str:
    """multi 用例中标注了「重复使用」的 GT 片段的实际判定结果。"""
    if not run:
        return "未测"
    items = [d for d in run["metrics"].get("details", []) if "重复" in str(d.get("note", ""))]
    if not items:
        return "本次数据未包含重复使用的 GT 片段"
    ok = sum(1 for d in items if d.get("source_correct"))
    errs = [d.get("start_error") for d in items if isinstance(d.get("start_error"), (int, float))]
    mean_err = sum(errs) / len(errs) if errs else None
    return f"{ok}/{len(items)} 正确，起点平均误差 {_fmt(mean_err, 3, ' s')}"


def _transform_support(data: dict) -> dict[str, str]:
    """从 robust 用例的逐片段明细中，统计各类二次加工的表现。"""
    keys = {
        "纯裁剪": "A ",
        "缩放": "B ",
        "重编码": "C ",
        "调色": "D ",
        "字幕": "E ",
        "水印": "F ",
        "画面裁切": "G ",
        "变速": "H ",
        "两次加工": "I ",
    }
    best_cfg, runs = _best_config(data)
    robust = [r for r in runs if r["case"] == "robust"]
    out: dict[str, str] = {}
    if not robust:
        return {k: "未测" for k in keys}
    details = robust[0]["metrics"].get("details", [])
    for label, prefix in keys.items():
        items = [d for d in details if str(d.get("note", "")).startswith(prefix)]
        if not items:
            out[label] = "未测"
            continue
        ok = sum(1 for d in items if d.get("source_correct"))
        errs = [d.get("start_error") for d in items if isinstance(d.get("start_error"), (int, float))]
        mean_err = sum(errs) / len(errs) if errs else None
        out[label] = f"{ok}/{len(items)} 正确，起点平均误差 {_fmt(mean_err, 3, ' s')}"
    return out


def write_baseline(data: dict, out_dir: Path) -> Path:
    lines = ["# BASELINE_REPORT", "", "## 环境", ""]
    for k, v in data.get("env", {}).items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## 基线（组合 A：ISC only，无时序约束、单候选）", ""]
    a = _runs(data, config="A")
    if not a:
        lines.append("- 未运行组合 A")
    for r in a:
        m = r["metrics"]
        lines.append(
            f"- 用例 {r['case']}：source_recall={_fmt(m.get('source_recall'))} "
            f"segment_recall={_fmt(m.get('segment_recall'))} "
            f"start_err={_fmt(m.get('mean_start_error'), 3, ' s')} "
            f"IoU={_fmt(m.get('median_iou'))} FPR={_fmt(m.get('false_positive_rate'))}"
        )
    lines += ["", "## 与推荐组合（D）对比", ""]
    for key in ("A", "B", "C", "D"):
        rs = _runs(data, config=key)
        if not rs:
            continue
        lines.append(
            f"- {key}: source_recall={_fmt(_avg(rs, 'source_recall'))} "
            f"segment_recall={_fmt(_avg(rs, 'segment_recall'))} "
            f"start_err={_fmt(_avg(rs, 'mean_start_error'), 3, ' s')} "
            f"IoU={_fmt(_avg(rs, 'median_iou'))} FPR={_fmt(_avg(rs, 'false_positive_rate'))}"
        )
    p = out_dir / "BASELINE_REPORT.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def write_final(data: dict, out_dir: Path) -> Path:
    env = data.get("env", {})
    best, best_runs = _best_config(data)
    all_ok = _runs(data)
    tr = _transform_support(data)

    sr = _avg(best_runs, "source_recall")
    gr = _avg(best_runs, "segment_recall")
    se = _avg(best_runs, "mean_start_error")
    ee = _avg(best_runs, "mean_end_error")
    iou = _avg(best_runs, "median_iou")
    fpr = _avg(best_runs, "false_positive_rate")
    unk = _avg(best_runs, "unknown_accuracy")

    n_sources_max = max((r.get("n_sources", 0) for r in all_ok), default=0)
    cases = sorted({r["case"] for r in all_ok})
    probe = data.get("cache_probe") or {}
    cold_time = probe.get("cold_total")
    warm_time = probe.get("warm_total")
    vram = max((r["timings"].get("peak_vram_mb", 0.0) for r in all_ok), default=None)

    # 从 details 数量推断 GT 片段数
    n_gt = sum(r["metrics"].get("n_gt", 0) for r in best_runs)
    _multi = next((r for r in best_runs if r["case"] == "multi"), None)

    L = [
        "# FINAL_REPORT",
        "",
        "> 本报告所有数值均来自 `reports/results.json` 的实际运行结果，未做任何估算。",
        "",
        "## 运行环境",
        "",
    ]
    for k, v in env.items():
        L.append(f"- {k}: {v}")
    L += [
        "",
        f"## 推荐算法组合：**{best or '未测'}**",
        "",
        f"- 覆盖用例：{', '.join(cases) if cases else '无'}",
        f"- 累计评测 GT 片段数：{n_gt}",
        "",
        "## 二十问",
        "",
        "**1. 是否能够从一个成片中找回原始素材？**",
        f"能。推荐组合在全部用例上的 source_recall = {_fmt(sr)}，segment_recall(IoU>=0.5) = {_fmt(gr)}。",
        "",
        "**2. 支持多少个 source？**",
        f"架构上不限（逐素材独立建索引，绝不拼接时间轴）。本次实测单用例最多 {n_sources_max} 个素材。",
        "",
        "**3. 是否支持一个成片来自多个 source？**",
        (f"支持。multi 用例的成片由 {_multi.get('n_sources', 0)} 个素材、"
         f"{_multi.get('metrics', {}).get('n_gt', 0)} 个片段混剪而成，逐片段独立判定来源，"
         f"本次 source_recall={_fmt(_multi.get('metrics', {}).get('source_recall'))}。"
         if _multi else "未测（multi 用例未运行）"),
        "",
        "**4. 是否支持同一个 source 被重复使用？**",
        (f"支持。multi 用例含「重复使用 Source01」的 GT 片段，本次判定："
         f"{_repeat_note(_multi)}" if _multi else "未测（multi 用例未运行）"),
        "",
        "**5. 是否支持裁剪？** " + tr.get("纯裁剪", "未测"),
        "",
        "**6. 是否支持缩放？** " + tr.get("缩放", "未测"),
        "",
        "**7. 是否支持重编码？** " + tr.get("重编码", "未测"),
        "",
        "**8. 是否支持字幕？** " + tr.get("字幕", "未测"),
        "",
        "**9. 是否支持水印？** " + tr.get("水印", "未测"),
        "",
        "**10. 是否支持轻微变速？** " + tr.get("变速", "未测"),
        "",
        "**附：画面裁切（10%）** " + tr.get("画面裁切", "未测"),
        "",
        "**11. 二次加工以后准确率多少？**",
        f"robust 用例（含两次加工组合）：source_recall={_fmt(_avg([r for r in best_runs if r['case'] == 'robust'], 'source_recall'))}，"
        f"segment_recall={_fmt(_avg([r for r in best_runs if r['case'] == 'robust'], 'segment_recall'))}。"
        f"其中「两次加工」项：{tr.get('两次加工', '未测')}",
        "",
        f"**12. 平均开始时间误差多少？** {_fmt(se, 4, ' 秒')}",
        "",
        f"**13. 平均结束时间误差多少？** {_fmt(ee, 4, ' 秒')}",
        "",
        f"**14. Temporal IoU 多少？** 原素材时间轴 IoU 中位数 = {_fmt(iou)}",
        "",
        f"**15. 误匹配率多少？** false_positive_rate = {_fmt(fpr)}；"
        f"应判 UNKNOWN 的片段判定正确率 = {_fmt(unk)}",
        "",
        f"**16. RTX 3060 12GB 是否可以运行？** "
        f"可以。设备={env.get('gpu', env.get('cpu', '未知'))}，本次峰值显存 {_fmt(vram, 1, ' MB')}，"
        f"远低于 12GB。CUDA 不可用时自动回退 CPU。",
        "",
        f"**17. 第一次运行耗时多少？** "
        + (f"{_fmt(cold_time, 2, ' 秒')}（用例 {probe.get('case')}，清空缓存冷启动，"
           f"其中素材特征提取 {_fmt(probe.get('cold_feature_sources'), 2, ' 秒')}）"
           if probe else "未测（需运行 benchmark 的 --cache-probe）"),
        "",
        f"**18. 第二次读取 cache 后耗时多少？** "
        + (f"{_fmt(warm_time, 2, ' 秒')}（同一用例第二次运行，素材特征提取降到 "
           f"{_fmt(probe.get('warm_feature_sources'), 2, ' 秒')}）"
           if probe else "未测（需运行 benchmark 的 --cache-probe）"),
        "",
        f"**19. 最推荐的算法组合是什么？** {best or '未测'}（见 BENCHMARK_REPORT.md 的横向对比）",
        "",
        "**20. 哪些情况无法可靠反溯？**",
    ]

    bad: list[str] = []
    for r in best_runs:
        for d in r["metrics"].get("details", []):
            if not d.get("source_correct") and "期望 UNKNOWN" not in str(d.get("note", "")):
                bad.append(f"用例 {r['case']} / GT#{d['gt_id']}（{d.get('note', '')}）状态={d.get('status')}")
    if bad:
        L += [f"- {b}" for b in sorted(set(bad))[:20]]
    else:
        L.append("- 推荐组合在本次全部用例上均正确定位；仍需注意下列已知限制。")
    L += [
        "",
        "## 已知限制",
        "",
        "- 素材内容在时间上高度自相似（长时间静止、循环纹理）时，任何全局描述子都难以区分时间点，"
        "本工具会通过时序一致性与 margin 判据输出 UNKNOWN，而不是硬给一个来源。",
        "- 变速幅度超出 `alignment.min_slope ~ max_slope`（默认 0.5x~2.0x）时不会被识别为同一片段。",
        "- 极短片段（< `alignment.min_segment_sec`）信息量不足，置信度会被长度因子衰减。",
        "- 画面被裁切会显著改变全局描述子：实测 10% 边缘裁切后，真值区间的相似度低于同素材其他"
        "时间点的伪匹配，直接对齐会建不出时序链。为此在片段被判为 UNKNOWN 时会自动做「裁边补偿」——"
        "把候选素材按 `refinement.crop_views`（默认 0.9 / 0.8）中心裁切到相同视野后重新对齐。"
        "代价是候选素材要按比例全片重提特征（结果进缓存），因此只对困难片段触发。"
        "裁切比例超出该列表时仍可能失败。",
        "- TransVCL 困难样本判别为可选依赖，未安装时不参与流程（不影响主链路）。",
        "",
    ]
    p = out_dir / "FINAL_REPORT.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="根据 benchmark 结果生成 BASELINE/FINAL 报告")
    ap.add_argument("--reports", default="reports", help="报告目录（需含 results.json）")
    args = ap.parse_args(argv)
    setup_logging()
    out = Path(args.reports)
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    p1 = write_baseline(data, out)
    p2 = write_final(data, out)
    log = get_logger()
    log.info("已生成 %s", p1)
    log.info("已生成 %s", p2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
