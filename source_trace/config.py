"""全局配置。

所有阈值 / 权重 / 采样率均可配置，禁止在算法代码里硬编码。
支持从 YAML/JSON 覆盖：``SourceTraceConfig.load(path)``。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".ts")


@dataclass
class SamplingConfig:
    """四级采样率：粗定位 -> 中密度补救 -> 候选精化 -> 困难样本。"""

    coarse_fps: float = 1.0
    # 粗定位（1FPS）与真实剪辑点最多相差 0.5s 相位，运动较快的素材上足以让 TN 找不到链。
    # 此时对候选素材以 medium_fps 全片重扫一次（结果进缓存），把相位差压到 0.25s。
    # 设为 <= coarse_fps 可关闭该补救。
    medium_fps: float = 2.0
    fine_fps: float = 4.0
    hard_fps: float = 8.0
    # 全片重扫的素材时长上限（秒）：超过该长度的素材不做高密度全片重扫，避免长片代价失控
    rescan_max_source_sec: float = 1200.0
    # 精定位时在候选区间两侧额外扩展的秒数
    fine_pad_sec: float = 1.5
    # 送入模型的短边分辨率（ISC21 官方 checkpoint 的 input_size=512）
    frame_size: int = 512


@dataclass
class ShotConfig:
    """镜头检测配置。"""

    # auto / pyscenedetect / histogram / fixed
    method: str = "auto"
    # 直方图差分阈值上限（0~1）：实际阈值 = min(该值, max(hist_min_abs, 中位数 + hist_z·MAD))
    hist_threshold: float = 0.35
    # 稳健阈值的 z 系数与绝对下限
    hist_z: float = 6.0
    hist_min_abs: float = 0.12
    # 切点得分至少要达到噪声中位数的多少倍
    hist_peak_ratio: float = 2.5
    # PySceneDetect ContentDetector 阈值
    content_threshold: float = 27.0
    min_shot_sec: float = 0.6
    # 超长镜头会被切分，避免一个 shot 跨越多个来源
    max_shot_sec: float = 12.0
    # method=fixed 时的固定窗口长度
    fixed_window_sec: float = 2.0


@dataclass
class FeatureConfig:
    """ISC21 descriptor 配置。"""

    model: str = "isc21"  # isc21 / isc21_lite
    backbone: str = "tf_efficientnetv2_m_in21ft1k"
    fallback_backbone: str = "tf_efficientnetv2_s_in21ft1k"
    dim: int = 256
    batch_size: int = 32
    device: str = "auto"  # auto / cuda / cpu
    fp16: bool = True
    # 权重文件（相对 models/ 目录或绝对路径）；为空则自动探测
    weights: str = ""
    # 权重下载地址（按顺序尝试；国内可用 ghfast.top 等 GitHub 加速前缀）
    weight_urls: tuple[str, ...] = (
        "https://ghfast.top/https://github.com/lyakaap/ISC21-Descriptor-Track-1st/releases/download/v1.0.1/isc_ft_v107.pth.tar",
        "https://github.com/lyakaap/ISC21-Descriptor-Track-1st/releases/download/v1.0.1/isc_ft_v107.pth.tar",
    )
    modality: str = "image"  # 严禁与 text embedding 混用
    normalize: str = "l2"


@dataclass
class CacheConfig:
    enabled: bool = True
    dir: str = "cache"
    full_hash: bool = False


@dataclass
class RetrievalConfig:
    top_k: int = 5
    # 单帧检索时每帧取的最近邻个数
    frame_top_n: int = 20
    # 候选 source 的最低平均相似度，低于该值直接剔除
    min_source_sim: float = 0.30
    backend: str = "auto"  # auto / faiss / numpy


@dataclass
class AlignmentConfig:
    """VCSL 时序对齐配置。"""

    method: str = "tn"  # tn / dp / dtw / hv / none
    # 相似度矩阵二值化阈值（TN 构图用）
    min_sim: float = 0.35
    # 自适应阈值：不同描述子的相似度分布差异很大（如颜色直方图整体偏高），
    # 因此在 min_sim 之外，再按「只保留全矩阵前 keep_mult*Q 个最高分」求一个下限
    adaptive_min_sim: bool = True
    adaptive_keep_mult: float = 5.0
    # TN：允许的最大时间跳步
    tn_max_step: int = 10
    # TN：每帧保留的候选边数量
    tn_top_k: int = 5
    # 一条路径至少包含的节点数
    min_length: int = 3
    # 允许的斜率范围（对应变速 0.5x~2.0x）
    min_slope: float = 0.5
    max_slope: float = 2.0
    # 允许的中断长度（秒）
    discontinue_sec: float = 2.0
    # DP 相关
    dp_gap_penalty: float = 0.2
    # 输出片段最短时长
    min_segment_sec: float = 0.8


@dataclass
class ConfidenceConfig:
    """最终 confidence 的加权组合，权重必须配置化。"""

    w_visual: float = 0.45
    w_temporal: float = 0.25
    w_margin: float = 0.15
    w_alignment: float = 0.15
    # 状态阈值
    high: float = 0.80
    medium: float = 0.65
    low: float = 0.50
    # 与第二候选的最小差距，过小则视为歧义
    ambiguous_margin: float = 0.04
    # 低于该置信度触发 TTA / TransVCL 增强
    enhance_threshold: float = 0.85
    # 片段长度归一化参考（秒）
    length_ref_sec: float = 3.0


@dataclass
class ExportConfig:
    # copy / reencode / auto
    mode: str = "auto"
    crf: int = 16
    preset: str = "slow"
    # 是否导出成片片段（用于对照）
    export_query_clips: bool = True
    # 是否把定位出来的原始片段按成片顺序拼成一条新素材 recovered_from_sources.mp4
    merge_recovered: bool = True
    # 可视化
    contact_sheet: bool = True
    compare_video: bool = True
    sheet_columns: int = 5
    sheet_thumb_width: int = 320


@dataclass
class RefinementConfig:
    enabled: bool = True
    # 是否启用 TTA（水平翻转 / 多尺度）
    tta: bool = False
    tta_scales: tuple[float, ...] = (1.0, 0.75)
    tta_hflip: bool = True
    # Score Normalization（借鉴 Meta AI Video Similarity Challenge 方案）
    # 默认关闭：属于「困难样本增强」，由 benchmark 组合 E 单独评估其收益
    score_norm: bool = False
    score_norm_k: int = 10
    # 裁边补偿：成片被裁掉画面边缘时，全局描述子会显著漂移。
    # 仅在片段仍判不出来（UNKNOWN / 置信度低于 confidence.low）时启用：
    # 把原素材也做中心裁切到相同视野再比对，逐个比例取最优。
    crop_compensate: bool = True
    crop_views: tuple[float, ...] = (0.9, 0.8)
    # 参与裁边补偿的候选素材个数（按检索排名截取，控制代价）
    crop_candidates: int = 3
    # TransVCL（可选依赖）
    transvcl: bool = False


@dataclass
class SourceTraceConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    shot: ShotConfig = field(default_factory=ShotConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)

    project_root: str = str(Path(__file__).resolve().parent.parent)

    # ---------- 序列化 ----------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str | None = None) -> "SourceTraceConfig":
        cfg = cls()
        if path is None:
            return cfg
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        cfg.update(data)
        return cfg

    def update(self, data: dict[str, Any]) -> None:
        """递归覆盖配置项，未知键会被忽略（并不静默——由调用方校验）。"""
        _apply(self, data)

    # ---------- 路径 ----------
    @property
    def models_dir(self) -> Path:
        return Path(self.project_root) / "models"

    def cache_dir(self, workdir: Path | str | None = None) -> Path:
        p = Path(self.cache.dir)
        if not p.is_absolute():
            base = Path(workdir) if workdir else Path(self.project_root)
            p = base / p
        return p


def _apply(obj: Any, data: dict[str, Any]) -> list[str]:
    unknown: list[str] = []
    valid = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in valid:
            unknown.append(key)
            continue
        cur = getattr(obj, key)
        if is_dataclass(cur) and isinstance(value, dict):
            unknown += [f"{key}.{u}" for u in _apply(cur, value)]
        else:
            if isinstance(cur, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(obj, key, value)
    return unknown
