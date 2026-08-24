"""命令行入口。

    python -m source_trace --query Final.mp4 --sources ./Sources --output ./output
    python reverse_trace.py Final.mp4 Sources/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import SourceTraceConfig
from .pipeline import run_trace
from .utils.log import get_logger, setup_logging
from .video.ffmpeg_tools import FFmpegNotFound, check_available


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m source_trace",
        description="成片原素材反溯工具：从混剪成片定位每个片段的原始素材与起止时间，并自动裁剪导出",
    )
    p.add_argument("query_pos", nargs="?", help="成片路径（位置参数写法）")
    p.add_argument("sources_pos", nargs="?", help="原始素材目录（位置参数写法）")
    p.add_argument("--query", "-q", help="成片路径")
    p.add_argument("--sources", "-s", help="原始素材目录")
    p.add_argument("--output", "-o", default="./output", help="输出目录（默认 ./output）")
    p.add_argument("--config", "-c", help="配置文件（yaml/json），覆盖默认配置")

    g = p.add_argument_group("算法")
    g.add_argument("--method", choices=["tn", "dp", "dtw", "hv", "none"], help="时序对齐方法（默认 tn）")
    g.add_argument("--coarse-fps", type=float, help="粗定位采样帧率（默认 1.0）")
    g.add_argument("--medium-fps", type=float, help="粗定位不自信时的全片重扫帧率（默认 2.0，设为 <= coarse 关闭）")
    g.add_argument("--fine-fps", type=float, help="精定位采样帧率（默认 4.0）")
    g.add_argument("--hard-fps", type=float, help="困难样本采样帧率（默认 8.0）")
    g.add_argument("--top-k", type=int, help="候选素材数量（默认 5）")
    g.add_argument("--frame-size", type=int, help="送入模型的帧尺寸（默认 512，与 ISC21 checkpoint 一致）")
    g.add_argument("--shot-method", choices=["auto", "pyscenedetect", "histogram", "fixed"], help="镜头检测方法")
    g.add_argument("--no-refine", action="store_true", help="关闭局部精定位")
    g.add_argument("--tta", action="store_true", help="困难样本启用 TTA")
    g.add_argument("--no-score-norm", action="store_true", help="关闭相似度归一化")

    g2 = p.add_argument_group("运行环境")
    g2.add_argument("--device", choices=["auto", "cuda", "cpu"], help="推理设备（默认 auto）")
    g2.add_argument("--batch-size", type=int, help="推理 batch 大小")
    g2.add_argument("--no-fp16", action="store_true", help="禁用 FP16")
    g2.add_argument("--model", choices=["isc21", "colorhist"], help="描述子模型（colorhist 为无 torch 兜底）")
    g2.add_argument("--weights", help="ISC 权重文件路径")
    g2.add_argument("--no-cache", action="store_true", help="禁用特征缓存")

    g3 = p.add_argument_group("输出")
    g3.add_argument("--no-clips", action="store_true", help="不导出裁剪片段")
    g3.add_argument("--no-merge", action="store_true", help="不把裁剪出的原始片段拼成新素材")
    g3.add_argument("--no-verify", action="store_true", help="不生成验证图/对照视频")
    g3.add_argument("--export-mode", choices=["auto", "copy", "reencode"], help="裁剪方式（默认 auto=精确重编码）")
    g3.add_argument("--quiet", action="store_true", help="只输出警告与错误")
    g3.add_argument("--debug", action="store_true", help="输出调试日志")
    return p


def _apply_args(cfg: SourceTraceConfig, a: argparse.Namespace) -> None:
    if a.method:
        cfg.alignment.method = a.method
    if a.coarse_fps:
        cfg.sampling.coarse_fps = a.coarse_fps
    if a.medium_fps:
        cfg.sampling.medium_fps = a.medium_fps
    if a.fine_fps:
        cfg.sampling.fine_fps = a.fine_fps
    if a.hard_fps:
        cfg.sampling.hard_fps = a.hard_fps
    if a.top_k:
        cfg.retrieval.top_k = a.top_k
    if a.frame_size:
        cfg.sampling.frame_size = a.frame_size
    if a.shot_method:
        cfg.shot.method = a.shot_method
    if a.no_refine:
        cfg.refinement.enabled = False
    if a.tta:
        cfg.refinement.tta = True
    if a.no_score_norm:
        cfg.refinement.score_norm = False
    if a.device:
        cfg.feature.device = a.device
    if a.batch_size:
        cfg.feature.batch_size = a.batch_size
    if a.no_fp16:
        cfg.feature.fp16 = False
    if a.model:
        cfg.feature.model = a.model
    if a.weights:
        cfg.feature.weights = a.weights
    if a.no_cache:
        cfg.cache.enabled = False
    if a.export_mode:
        cfg.export.mode = a.export_mode
    if a.no_merge:
        cfg.export.merge_recovered = False


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    query = args.query or args.query_pos
    sources = args.sources or args.sources_pos
    if not query or not sources:
        parser.print_help()
        print("\n错误：必须提供成片与素材目录。示例：")
        print("  python -m source_trace --query Final.mp4 --sources ./Sources --output ./output")
        print("  python reverse_trace.py Final.mp4 Sources/")
        return 2

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    level = logging.WARNING if args.quiet else (logging.DEBUG if args.debug else logging.INFO)
    log = setup_logging(level, out_dir / "logs" / "run.log")

    try:
        info = check_available()
        log.info("FFmpeg：%s", info["ffmpeg"])
        log.info("FFprobe：%s", info["ffprobe"])
    except FFmpegNotFound as exc:
        log.error("%s", exc)
        return 3

    cfg = SourceTraceConfig.load(args.config) if args.config else SourceTraceConfig()
    _apply_args(cfg, args)

    try:
        res = run_trace(
            query=query,
            sources=sources,
            output=out_dir,
            cfg=cfg,
            export_clips=not args.no_clips,
            verify=not args.no_verify,
        )
    except Exception as exc:
        get_logger().error("运行失败：%s: %s", type(exc).__name__, exc, exc_info=args.debug)
        return 1

    print()
    print(f"结果目录：{res.output_dir.resolve()}")
    print(f"  result.json / result.csv  共 {len(res.segments)} 个片段")
    for s in res.segments:
        if s.status == "UNKNOWN":
            print(f"  #{s.id:02d} {s.query_start:7.2f}-{s.query_end:7.2f}  ->  UNKNOWN（confidence={s.confidence:.3f}）")
        else:
            print(
                f"  #{s.id:02d} {s.query_start:7.2f}-{s.query_end:7.2f}  ->  {s.source}  "
                f"{s.source_start:7.2f}-{s.source_end:7.2f}  conf={s.confidence:.3f} [{s.status}]"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
