"""自动评价指标。

指标定义（严格、可复现）：

* ``source_recall``    有多少 GT 片段的来源素材被正确识别（按成片时间轴重叠配对）
* ``segment_recall``   在来源正确的前提下，原素材时间区间 IoU >= iou_thr 的比例
* ``mean_start_error`` |预测 source_start - GT source_start| 的均值（秒）
* ``mean_end_error``   |预测 source_end - GT source_end| 的均值（秒）
* ``median_iou``       原素材时间轴 1D IoU 的中位数
* ``median_iou_2d``    (query, source) 二维框 IoU 中位数（VCSL 风格）
* ``false_positive_rate`` 给出了来源但来源错误 / 无对应 GT 的预测比例
* ``unknown_accuracy``    应当 UNKNOWN 的片段中真的输出 UNKNOWN 的比例

配对规则：预测片段与 GT 片段按成片时间轴重叠时长最大者配对，且重叠比例需 >= 0.3。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .ground_truth import GroundTruth, GTSegment, PredSegment


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _iou_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = _overlap(a0, a1, b0, b1)
    union = (a1 - a0) + (b1 - b0) - inter
    return float(inter / union) if union > 1e-9 else 0.0


def _iou_2d(pred: PredSegment, gt: GTSegment) -> float:
    if pred.source_start is None or pred.source_end is None:
        return 0.0
    iw = _overlap(pred.query_start, pred.query_end, gt.query_start, gt.query_end)
    ih = _overlap(pred.source_start, pred.source_end, gt.source_start, gt.source_end)
    inter = iw * ih
    a = (pred.query_end - pred.query_start) * (pred.source_end - pred.source_start)
    b = (gt.query_end - gt.query_start) * (gt.source_end - gt.source_start)
    union = a + b - inter
    return float(inter / union) if union > 1e-9 else 0.0


@dataclass
class MatchDetail:
    gt_id: int
    pred_id: int | None
    gt_source: str
    pred_source: str | None
    source_correct: bool
    start_error: float | None
    end_error: float | None
    aligned_start_error: float | None
    iou: float
    iou_2d: float
    status: str
    confidence: float
    note: str = ""


@dataclass
class EvalResult:
    source_recall: float
    segment_recall: float
    mean_start_error: float
    mean_end_error: float
    median_start_error: float
    median_end_error: float
    mean_aligned_start_error: float
    median_iou: float
    median_iou_2d: float
    false_positive_rate: float
    unknown_accuracy: float
    n_gt: int
    n_pred: int
    n_pred_named: int
    iou_threshold: float
    details: list[MatchDetail] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["details"] = [asdict(x) for x in self.details]
        return d

    def summary_lines(self) -> list[str]:
        return [
            f"source_recall        = {self.source_recall:.4f}",
            f"segment_recall(IoU>={self.iou_threshold:.2f}) = {self.segment_recall:.4f}",
            f"mean_start_error     = {self.mean_start_error:.4f} s",
            f"mean_end_error       = {self.mean_end_error:.4f} s",
            f"median_start_error   = {self.median_start_error:.4f} s",
            f"median_end_error     = {self.median_end_error:.4f} s",
            f"median_iou(source)   = {self.median_iou:.4f}",
            f"median_iou_2d        = {self.median_iou_2d:.4f}",
            f"false_positive_rate  = {self.false_positive_rate:.4f}",
            f"unknown_accuracy     = {self.unknown_accuracy:.4f}",
        ]


def evaluate(
    gt: GroundTruth,
    preds: list[PredSegment],
    iou_threshold: float = 0.5,
    min_query_overlap_ratio: float = 0.3,
) -> EvalResult:
    """对一次运行结果做全指标评估。"""
    details: list[MatchDetail] = []
    used_pred: set[int] = set()

    for g in gt.segments:
        best: tuple[float, PredSegment | None] = (0.0, None)
        for p in preds:
            ov = _overlap(p.query_start, p.query_end, g.query_start, g.query_end)
            ratio = ov / max(g.query_end - g.query_start, 1e-9)
            if ratio >= min_query_overlap_ratio and ov > best[0]:
                best = (ov, p)
        p = best[1]

        if p is None:
            details.append(
                MatchDetail(g.id, None, g.source, None, False, None, None, None, 0.0, 0.0, "MISS", 0.0, g.note)
            )
            continue
        used_pred.add(p.id)

        if g.is_unknown_expected:
            correct_unknown = p.status == "UNKNOWN" or p.source is None
            details.append(
                MatchDetail(
                    g.id, p.id, g.source, p.source, correct_unknown, None, None, None,
                    0.0, 0.0, p.status, p.confidence, g.note + "（期望 UNKNOWN）",
                )
            )
            continue

        source_correct = bool(p.source == g.source and p.status != "UNKNOWN")
        if source_correct and p.source_start is not None and p.source_end is not None:
            start_err = abs(p.source_start - g.source_start)
            end_err = abs(p.source_end - g.source_end)
            # 对齐误差：把预测的线性映射外推到 GT 的成片起点，剔除镜头切分差异的影响
            mapped = p.source_start + (g.query_start - p.query_start) * (p.speed or 1.0)
            aligned_err = abs(mapped - g.source_start)
            iou = _iou_1d(p.source_start, p.source_end, g.source_start, g.source_end)
            iou2 = _iou_2d(p, g)
        else:
            start_err = end_err = aligned_err = None
            iou = iou2 = 0.0

        details.append(
            MatchDetail(
                g.id, p.id, g.source, p.source, source_correct,
                start_err, end_err, aligned_err, iou, iou2, p.status, p.confidence, g.note,
            )
        )

    real = [d for d in details if "期望 UNKNOWN" not in d.note]
    unknown_expected = [d for d in details if "期望 UNKNOWN" in d.note]

    n_gt = len(real)
    src_ok = [d for d in real if d.source_correct]
    source_recall = len(src_ok) / n_gt if n_gt else 0.0
    seg_ok = [d for d in src_ok if d.iou >= iou_threshold]
    segment_recall = len(seg_ok) / n_gt if n_gt else 0.0

    starts = [d.start_error for d in src_ok if d.start_error is not None]
    ends = [d.end_error for d in src_ok if d.end_error is not None]
    aligned = [d.aligned_start_error for d in src_ok if d.aligned_start_error is not None]
    ious = [d.iou for d in src_ok]
    ious2 = [d.iou_2d for d in src_ok]

    named_preds = [p for p in preds if p.source is not None and p.status != "UNKNOWN"]
    correct_pred_ids = {d.pred_id for d in src_ok}
    fp = [p for p in named_preds if p.id not in correct_pred_ids]
    fpr = len(fp) / len(named_preds) if named_preds else 0.0

    unk_acc = (
        sum(1 for d in unknown_expected if d.source_correct) / len(unknown_expected)
        if unknown_expected
        else 1.0
    )

    def _m(xs, fn=np.mean):
        return float(fn(xs)) if xs else -1.0

    return EvalResult(
        source_recall=source_recall,
        segment_recall=segment_recall,
        mean_start_error=_m(starts),
        mean_end_error=_m(ends),
        median_start_error=_m(starts, np.median),
        median_end_error=_m(ends, np.median),
        mean_aligned_start_error=_m(aligned),
        median_iou=_m(ious, np.median),
        median_iou_2d=_m(ious2, np.median),
        false_positive_rate=fpr,
        unknown_accuracy=unk_acc,
        n_gt=n_gt,
        n_pred=len(preds),
        n_pred_named=len(named_preds),
        iou_threshold=iou_threshold,
        details=details,
    )
