"""合成测试数据生成（Synthetic Ground Truth）。

完全离线：使用 numpy + OpenCV 渲染帧，经 FFmpeg 编码为视频。不下载任何素材。

生成物：
    <out>/Sources/Source01.mp4 ... SourceNN.mp4
    <out>/Final.mp4
    <out>/ground_truth.json

每个 source 的视觉内容满足：
* 不同 source 之间风格/色调/图形明显不同（可区分）
* 同一 source 内每一帧唯一（含帧计数图案 + 时间文字），因此时间定位可被严格验证
* 可选生成「干扰素材」：与另一 source 风格极为接近，用于测试误匹配

Final 的每个片段可施加二次加工：裁剪、缩放、重编码、调色、字幕、水印、变速、二次加工叠加。
片段在成片时间轴上的位置由实际导出时长累加得到（而非理论值），因此 ground truth 精确。
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..utils.log import get_logger, setup_logging
from ..video.ffmpeg_tools import ffmpeg_path
from ..video.probe import probe

# ---------------------------------------------------------------- 渲染


_PALETTES = [
    # (基础色调 hue 0-179, 形状类型, 背景纹理)
    (5, "circle", "stripes"),
    (35, "rect", "checker"),
    (70, "triangle", "rings"),
    (105, "circle", "checker"),
    (140, "rect", "stripes"),
    (165, "triangle", "waves"),
    (20, "rect", "rings"),
    (90, "triangle", "stripes"),
]


_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _grids(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    key = (h, w)
    if key not in _GRID_CACHE:
        yy, xx = np.mgrid[0:h, 0:w]
        _GRID_CACHE[key] = (yy.astype(np.float32), xx.astype(np.float32))
    return _GRID_CACHE[key]


def _background(h: int, w: int, hue: int, texture: str, t: float) -> np.ndarray:
    yy, xx = _grids(h, w)
    if texture == "stripes":
        pat = np.sin((xx * 0.05) + t * 1.7) * 0.5 + 0.5
    elif texture == "checker":
        pat = ((np.floor(xx / 40 + t * 0.8).astype(np.int32) + np.floor(yy / 40).astype(np.int32)) % 2).astype(
            np.float32
        )
    elif texture == "rings":
        cx, cy = w / 2 + 60 * np.sin(t * 0.6), h / 2 + 40 * np.cos(t * 0.5)
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        pat = np.sin(r * 0.06 - t * 2.0) * 0.5 + 0.5
    else:  # waves
        pat = np.sin(xx * 0.03 + np.sin(yy * 0.02 + t) * 2.0) * 0.5 + 0.5

    hsv = np.empty((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = np.clip(hue + pat * 18.0 + (t * 3.0) % 12.0, 0, 179).astype(np.uint8)
    hsv[..., 1] = np.clip(120 + pat * 100, 0, 255).astype(np.uint8)
    hsv[..., 2] = np.clip(70 + pat * 150, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _draw_shapes(img: np.ndarray, rng: random.Random, shape: str, t: float, params: list[tuple]) -> None:
    h, w = img.shape[:2]
    for (ax, ay, bx, by, size, cb, cg, cr, spin) in params:
        cx = int((0.5 + 0.42 * np.sin(ax * t + bx)) * w)
        cy = int((0.5 + 0.40 * np.cos(ay * t + by)) * h)
        color = (int(cb), int(cg), int(cr))
        if shape == "circle":
            cv2.circle(img, (cx, cy), int(size), color, -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (cx, cy), int(size), (255, 255, 255), 2, lineType=cv2.LINE_AA)
        elif shape == "rect":
            ang = spin * t
            box = cv2.boxPoints(((cx, cy), (size * 2, size * 1.4), np.degrees(ang)))
            cv2.fillConvexPoly(img, box.astype(np.int32), color, lineType=cv2.LINE_AA)
        else:  # triangle
            ang = spin * t
            pts = np.array(
                [
                    [cx + size * np.cos(ang + k * 2.094), cy + size * np.sin(ang + k * 2.094)]
                    for k in range(3)
                ],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(img, pts, color, lineType=cv2.LINE_AA)


def _draw_frame_code(img: np.ndarray, idx: int) -> None:
    """在画面右下角绘制帧序号二进制码块，保证每帧图像唯一。"""
    h, w = img.shape[:2]
    cell = max(6, w // 80)
    x0, y0 = w - cell * 18 - 8, h - cell - 8
    for b in range(18):
        on = (idx >> b) & 1
        c = (255, 255, 255) if on else (0, 0, 0)
        cv2.rectangle(img, (x0 + b * cell, y0), (x0 + (b + 1) * cell - 1, y0 + cell - 1), c, -1)


def _make_strip(h: int, w_total: int, rng: random.Random, tile: int = 48) -> np.ndarray:
    """生成一条超宽的随机色块长条，用于横向平移。

    这是让「同一素材的不同时刻」在全局视觉统计上明显不同的关键：
    若素材内容随时间几乎不变（例如只有周期性纹理），任何全局描述子都无法区分时间，
    合成数据就失去了评测意义。
    """
    rows = int(np.ceil(h / tile))
    cols = int(np.ceil(w_total / tile))
    small = np.zeros((rows, cols, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            small[r, c] = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    strip = np.repeat(np.repeat(small, tile, axis=0), tile, axis=1)
    return strip[:h, :w_total]


def render_source(
    out_path: Path,
    index: int,
    duration: float,
    size: tuple[int, int] = (854, 480),
    fps: float = 25.0,
    seed: int | None = None,
    style_index: int | None = None,
    label: str | None = None,
    crf: int = 18,
) -> None:
    """渲染一个原始素材视频。"""
    log = get_logger()
    w, h = size
    seed = index * 7919 if seed is None else seed
    rng = random.Random(seed)
    hue, shape, texture = _PALETTES[(index - 1 if style_index is None else style_index) % len(_PALETTES)]
    label = label or f"S{index:02d}"

    shape_params = [
        (
            rng.uniform(0.3, 1.2),
            rng.uniform(0.3, 1.2),
            rng.uniform(0, 6.28),
            rng.uniform(0, 6.28),
            rng.uniform(w * 0.03, w * 0.09),
            rng.randint(0, 255),
            rng.randint(0, 255),
            rng.randint(0, 255),
            rng.uniform(-2.0, 2.0),
        )
        for _ in range(4)
    ]

    # 横向平移的随机色块长条：保证内容随时间持续变化且不重复
    pan_speed = w / 18.0  # 约 18 秒滚过一屏
    strip = _make_strip(h, int(w + pan_speed * duration) + 64, random.Random(seed + 1))

    n_frames = int(round(duration * fps))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}",
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-g", "50",
        str(out_path),
    ]
    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for i in range(n_frames):
            t = i / fps
            bg = _background(h, w, hue, texture, t)
            x0 = int(pan_speed * t)
            window = strip[:, x0 : x0 + w]
            img = cv2.addWeighted(window, 0.62, bg, 0.38, 0)
            _draw_shapes(img, rng, shape, t, shape_params)
            cv2.putText(img, label, (16, 52), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(img, label, (16, 52), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                img, f"t={t:07.3f}", (16, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA
            )
            cv2.putText(
                img, f"t={t:07.3f}", (16, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA
            )
            _draw_frame_code(img, i)
            proc.stdin.write(img.tobytes())
    finally:
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "ignore") if proc.stderr else ""
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"渲染 {out_path.name} 失败：{err[-800:]}")
    log.info("已生成素材 %s（%.1fs, %dx%d@%.0ffps, 风格=%s/%s）", out_path.name, duration, w, h, fps, shape, texture)


# ---------------------------------------------------------------- 加工


@dataclass
class SegmentPlan:
    """一个成片片段的生成计划。"""

    source: str  # 素材文件名
    source_start: float
    source_end: float
    # 加工项
    scale: float = 1.0  # 缩放比例
    crop: float = 0.0  # 裁边比例（0.1 表示四周各裁 10%）
    speed: float = 1.0  # 速度倍率（>1 加速）
    color: bool = False  # 调色
    subtitle: bool = False  # 字幕
    watermark: bool = False  # 水印
    reencode_twice: bool = False  # 二次加工（低码率再压一次）
    note: str = ""


def _filters_for(plan: SegmentPlan, w: int, h: int) -> list[str]:
    f: list[str] = []
    if plan.crop > 0:
        cw, ch = 1.0 - 2 * plan.crop, 1.0 - 2 * plan.crop
        f.append(f"crop=iw*{cw:.4f}:ih*{ch:.4f}:iw*{plan.crop:.4f}:ih*{plan.crop:.4f}")
    if plan.scale != 1.0:
        f.append(f"scale=trunc(iw*{plan.scale:.4f}/2)*2:trunc(ih*{plan.scale:.4f}/2)*2")
    if plan.color:
        f.append("eq=brightness=0.08:contrast=1.25:saturation=1.45,hue=h=12")
    if plan.watermark:
        f.append("drawbox=x=iw-iw/4:y=8:w=iw/4-8:h=ih/12:color=white@0.55:t=fill")
    if plan.subtitle:
        f.append("drawbox=x=0:y=ih-ih/8:w=iw:h=ih/8:color=black@0.6:t=fill")
    if plan.speed != 1.0:
        f.append(f"setpts=PTS/{plan.speed:.6f}")
    return f


def _export_clip(src: Path, plan: SegmentPlan, out: Path, base_size: tuple[int, int], fps: float) -> None:
    dur = plan.source_end - plan.source_start
    vf = _filters_for(plan, *base_size)
    # 统一回到基准分辨率与帧率，保证 concat 可行
    vf.append(f"scale={base_size[0]}:{base_size[1]}:flags=bicubic")
    vf.append(f"fps={fps}")
    args = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{plan.source_start:.6f}", "-i", str(src), "-t", f"{dur:.6f}",
        "-an", "-vf", ",".join(vf),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        str(out),
    ]
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"导出片段失败 {out.name}: {proc.stderr.decode('utf-8','ignore')[-800:]}")

    if plan.reencode_twice:
        tmp = out.with_name(out.stem + "_2nd.mp4")
        args2 = [
            ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(out), "-an",
            "-vf", f"scale=trunc(iw*0.7/2)*2:trunc(ih*0.7/2)*2,scale={base_size[0]}:{base_size[1]}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "31", "-pix_fmt", "yuv420p",
            str(tmp),
        ]
        p2 = subprocess.run(args2, capture_output=True)
        if p2.returncode != 0:
            raise RuntimeError(f"二次加工失败 {out.name}: {p2.stderr.decode('utf-8','ignore')[-800:]}")
        out.unlink()
        tmp.rename(out)


def build_final(
    sources_dir: Path,
    plans: list[SegmentPlan],
    out_final: Path,
    workdir: Path,
    base_size: tuple[int, int] = (854, 480),
    fps: float = 25.0,
) -> dict:
    """按计划导出各片段并拼接为成片，返回 ground truth。"""
    log = get_logger()
    workdir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    for i, plan in enumerate(plans, 1):
        clip = workdir / f"clip_{i:03d}.mp4"
        _export_clip(sources_dir / plan.source, plan, clip, base_size, fps)
        clips.append(clip)

    listfile = workdir / "concat.txt"
    listfile.write_text(
        "\n".join(f"file '{c.resolve().as_posix()}'" for c in clips) + "\n", encoding="utf-8"
    )
    out_final.parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(listfile),
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr", "-r", f"{fps}",
        str(out_final),
    ]
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"拼接成片失败：{proc.stderr.decode('utf-8','ignore')[-1000:]}")

    # 用实际片段时长累加得到成片时间轴（精确 ground truth）
    cursor = 0.0
    gt_segments = []
    for i, (plan, clip) in enumerate(zip(plans, clips), 1):
        d = probe(clip).duration
        gt_segments.append(
            {
                "id": i,
                "query_start": round(cursor, 4),
                "query_end": round(cursor + d, 4),
                "source": plan.source,
                "source_start": round(plan.source_start, 4),
                "source_end": round(plan.source_end, 4),
                "speed": plan.speed,
                "transforms": {
                    k: v
                    for k, v in asdict(plan).items()
                    if k not in ("source", "source_start", "source_end", "note")
                    and v not in (0.0, 1.0, False, "")
                },
                "note": plan.note,
            }
        )
        cursor += d

    final_info = probe(out_final)
    gt = {
        "query": out_final.name,
        "query_duration": round(final_info.duration, 4),
        "sources": sorted(p.name for p in sources_dir.glob("*.mp4")),
        "segments": gt_segments,
    }
    log.info("已生成成片 %s（%.2fs，%d 个片段）", out_final.name, final_info.duration, len(gt_segments))
    return gt


# ---------------------------------------------------------------- 预设用例


@dataclass
class CaseSpec:
    name: str
    n_sources: int
    source_duration: float
    plans: list[SegmentPlan] = field(default_factory=list)
    # 干扰素材：{索引: 复制风格的源索引}
    distractors: dict[int, int] = field(default_factory=dict)
    size: tuple[int, int] = (854, 480)
    fps: float = 25.0


def _case_basic() -> CaseSpec:
    """基础：3 个素材，纯裁剪无加工。"""
    return CaseSpec(
        name="basic",
        n_sources=3,
        source_duration=60.0,
        plans=[
            SegmentPlan("Source03.mp4", 52.0, 56.0, note="纯裁剪"),
            SegmentPlan("Source01.mp4", 17.5, 22.0, note="纯裁剪"),
            SegmentPlan("Source02.mp4", 31.0, 35.5, note="纯裁剪"),
        ],
    )


def _case_multi() -> CaseSpec:
    """多素材 + 重复使用同一 source + 短片段 + 大量无关内容。"""
    return CaseSpec(
        name="multi",
        n_sources=5,
        source_duration=90.0,
        plans=[
            SegmentPlan("Source03.mp4", 52.31, 56.42, note="纯裁剪"),
            SegmentPlan("Source01.mp4", 17.32, 21.91, note="纯裁剪"),
            SegmentPlan("Source05.mp4", 31.82, 35.55, note="纯裁剪"),
            SegmentPlan("Source02.mp4", 70.00, 71.20, note="短片段 1.2s"),
            SegmentPlan("Source01.mp4", 60.50, 65.00, note="重复使用 Source01"),
            SegmentPlan("Source04.mp4", 5.00, 9.40, note="纯裁剪"),
        ],
    )


def _case_robust() -> CaseSpec:
    """二次加工鲁棒性：每种加工单独占一个片段，便于逐项统计支持度。

    A 纯裁剪 / B 缩放 / C 缩放+重编码 / D 调色 / E 字幕 / F 水印 / G 画面裁切 /
    H 变速 1.25x / H 变速 0.9x / I 两次加工组合。
    """
    return CaseSpec(
        name="robust",
        n_sources=5,
        source_duration=90.0,
        plans=[
            SegmentPlan("Source01.mp4", 10.0, 14.5, note="A 纯裁剪"),
            SegmentPlan("Source02.mp4", 22.0, 26.5, scale=0.5, note="B 裁剪+缩放"),
            SegmentPlan("Source03.mp4", 40.0, 44.5, scale=0.6, note="C 裁剪+缩放+重编码"),
            SegmentPlan("Source04.mp4", 55.0, 59.5, color=True, note="D 裁剪+调色+重编码"),
            SegmentPlan("Source05.mp4", 12.0, 16.5, subtitle=True, note="E 字幕"),
            SegmentPlan("Source05.mp4", 30.0, 34.5, watermark=True, note="F 水印"),
            SegmentPlan("Source04.mp4", 20.0, 24.5, crop=0.10, note="G 画面裁切 10%"),
            SegmentPlan("Source01.mp4", 70.0, 75.0, speed=1.25, note="H 变速 1.25x"),
            SegmentPlan("Source02.mp4", 50.0, 55.0, speed=0.9, note="H 变速 0.9x"),
            SegmentPlan(
                "Source03.mp4", 5.0, 9.5, crop=0.12, color=True, subtitle=True, reencode_twice=True,
                note="I 两次加工组合",
            ),
        ],
    )


def _case_distractor() -> CaseSpec:
    """干扰素材：Source04/Source05 与 Source01/Source02 风格完全相同（仅内容序列不同）。"""
    spec = CaseSpec(
        name="distractor",
        n_sources=5,
        source_duration=70.0,
        distractors={4: 1, 5: 2},
        plans=[
            SegmentPlan("Source01.mp4", 20.0, 24.0, note="与 Source04 同风格"),
            SegmentPlan("Source02.mp4", 35.0, 39.0, note="与 Source05 同风格"),
            SegmentPlan("Source04.mp4", 50.0, 54.0, note="干扰素材本身作为来源"),
            SegmentPlan("Source03.mp4", 10.0, 14.0, note="独立风格"),
        ],
    )
    return spec


def _case_unknown() -> CaseSpec:
    """含无来源片段：一个片段来自未提供的素材，应输出 UNKNOWN。"""
    return CaseSpec(
        name="unknown",
        n_sources=3,
        source_duration=60.0,
        plans=[
            SegmentPlan("Source01.mp4", 12.0, 16.0, note="有来源"),
            SegmentPlan("_Hidden.mp4", 20.0, 24.0, note="来源被移除，期望 UNKNOWN"),
            SegmentPlan("Source03.mp4", 40.0, 44.0, note="有来源"),
        ],
    )


CASES = {
    "basic": _case_basic,
    "multi": _case_multi,
    "robust": _case_robust,
    "distractor": _case_distractor,
    "unknown": _case_unknown,
}


def _already_rendered(path: Path, expect_duration: float) -> bool:
    """素材已存在且时长匹配时可复用，避免重复渲染。"""
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        return abs(probe(path).duration - expect_duration) < 0.5
    except Exception:
        return False


def make_dataset(out_dir: Path | str, case: str = "basic", force: bool = False) -> Path:
    """生成一套测试数据；返回数据集根目录。"""
    log = get_logger()
    if case not in CASES:
        raise ValueError(f"未知用例 {case}，可选：{sorted(CASES)}")
    spec = CASES[case]()
    root = Path(out_dir)
    sources_dir = root / "Sources"
    final_path = root / "Final.mp4"
    gt_path = root / "ground_truth.json"

    if gt_path.exists() and not force:
        log.info("测试数据已存在，跳过生成：%s（--force 可强制重建）", root)
        return root

    sources_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, spec.n_sources + 1):
        style = spec.distractors.get(i)
        dst = sources_dir / f"Source{i:02d}.mp4"
        if not force and _already_rendered(dst, spec.source_duration):
            log.info("素材已存在，复用：%s", dst.name)
            continue
        render_source(
            dst,
            index=i,
            duration=spec.source_duration,
            size=spec.size,
            fps=spec.fps,
            seed=i * 7919 + (0 if style is None else 13),
            style_index=(style - 1) if style else None,
            label=f"S{i:02d}",
        )

    # unknown 用例需要一个「隐藏素材」，生成后不放入 Sources
    hidden_dir = root / "_hidden"
    if any(p.source.startswith("_Hidden") for p in spec.plans):
        hidden_dir.mkdir(parents=True, exist_ok=True)
        hid = hidden_dir / "_Hidden.mp4"
        if force or not _already_rendered(hid, spec.source_duration):
            render_source(
                hid,
                index=spec.n_sources + 1,
                duration=spec.source_duration,
                size=spec.size,
                fps=spec.fps,
                seed=99991,
                label="HID",
            )

    workdir = root / "_clips"

    class _Resolver:
        """让 build_final 能同时从 Sources/ 和 _hidden/ 取素材。"""

        def __truediv__(self, name: str) -> Path:
            p = sources_dir / name
            return p if p.exists() else hidden_dir / name

        def glob(self, pattern: str):
            return sources_dir.glob(pattern)

    gt = build_final(_Resolver(), spec.plans, final_path, workdir, base_size=spec.size, fps=spec.fps)  # type: ignore[arg-type]
    gt["case"] = spec.name
    gt_path.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")

    # 清理中间片段
    for p in workdir.glob("*"):
        p.unlink()
    workdir.rmdir()

    log.info("测试数据集就绪：%s（case=%s，%d 素材）", root, spec.name, spec.n_sources)
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成合成测试数据（Sources + Final + ground_truth.json）")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--case", default="basic", choices=sorted(CASES), help="测试用例")
    parser.add_argument("--force", action="store_true", help="已存在时强制重建")
    args = parser.parse_args(argv)
    setup_logging()
    make_dataset(args.out, args.case, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
