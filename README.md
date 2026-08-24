# 成片原素材反溯工具 / Source Trace

从一个**已经混剪完成的成片**反向定位它的每一个片段来自哪个**原始素材**、
在原始素材中的**准确起止时间**，判断可信度，并自动把对应的原始片段裁剪导出。

这不是视频相似度检测，也不是视频内容理解，而是 **Video Copy Localization**：

```
Final.mp4 + Sources/(N 个素材)
        ↓
Shot Detection → ISC21 256D descriptor → 多素材 Top-K 检索
        ↓
VCSL / Temporal Network 时序对齐 → 局部高帧率精定位 → 置信度分级
        ↓
result.json / result.csv / 原始片段 mp4 / 验证图
```

---

## 1. 快速开始

```bash
# 安装依赖（PyTorch 需按显卡单独装，见下）
python -m pip install -r requirements.txt

# 反溯
python -m source_trace --query Final.mp4 --sources ./Sources --output ./output
# 或
python reverse_trace.py Final.mp4 Sources/

# 图形界面
python -m source_trace.gui
```

输出：

```
output/
├── result.json           # 完整结果（含候选、各项子分数、耗时）
├── result.csv            # 表格结果
├── errors.json           # 出错的素材/片段（若有）
├── logs/
│   └── run.log           # 中文运行日志
├── segments/             # 自动裁剪出的原始片段（+ 对应的成片片段）
│   ├── 001_Source03_00-52-31_00-56-42.mp4
│   └── 002_Source01_01-17-32_01-21-91.mp4
└── verification/         # 人工验证材料
    ├── segment_001.jpg           # 上：成片抽帧 / 下：原素材抽帧
    └── segment_001_compare.mp4   # 左右并排对照视频
```

### PyTorch 安装

```bash
# CUDA 12.4（RTX 3060 等，推荐）
python -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    -f https://mirrors.aliyun.com/pytorch-wheels/cu124/

# 仅 CPU
python -m pip install torch torchvision
```

> Windows 的 PyPI 版 torch 是 **CPU-only**，装 CUDA 版必须用上面的 wheel 源。

### FFmpeg

必须有 `ffmpeg` 与 `ffprobe`。三种方式任选：

1. 放到项目内 `tools/ffmpeg/`（本项目默认查找位置）
2. 加入系统 `PATH`
3. 设置环境变量 `SOURCE_TRACE_FFMPEG` / `SOURCE_TRACE_FFPROBE`

### ISC21 预训练权重

首次运行会自动尝试下载到 `models/`。也可手动下载：

```
https://github.com/lyakaap/ISC21-Descriptor-Track-1st/releases/download/v1.0.1/isc_ft_v107.pth.tar
```

放到 `models/isc_ft_v107.pth.tar` 即可（400 MB）。
**权重缺失时不会中断**：自动降级为 ImageNet 预训练 backbone + GeM；
连 PyTorch 都没有时，再降级为纯 numpy 的 `colorhist` 描述子。基础链路始终可运行。

---

## 2. 算法链路

| 阶段 | 模块 | 说明 |
| --- | --- | --- |
| 视频读取 | `video/probe.py` `video/reader.py` | ffprobe 取元数据；抽帧时间戳来自 FFmpeg `showinfo` 的真实 **pts**，对 VFR 同样正确 |
| 镜头检测 | `video/shot_detector.py` | PySceneDetect（可选）→ 直方图+像素差分（内置）→ 固定窗口，逐级降级 |
| 帧描述子 | `features/isc.py` `features/extractor.py` | ISC21：`tf_efficientnetv2_m_in21ft1k` → GeM(eval p=4) → Linear(512→256, no bias) → BN → L2 |
| 特征缓存 | `features/cache.py` | 键 = 文件内容指纹 + 参数指纹，源文件或参数变化自动失效 |
| 多素材检索 | `retrieval/` | 每个素材独立建索引（**绝不 concat 时间轴**），帧级 Top-K + 加权投票 |
| 时序对齐 | `temporal/alignment.py` | **TN**（Temporal Network，主力）/ DP / DTW / Hough Voting，纯 numpy 实现 |
| 时间映射 | `temporal/mapping.py` | 匹配点在**时间域**做鲁棒线性拟合 `source_t = k·query_t + b`，外推到镜头边界 |
| 局部精定位 | `refinement/fine_match.py` | 粗定位 1 FPS → 全片重扫 2/4 FPS（仅在粗定位不自信时，带缓存）→ 候选区间 4 FPS → 困难样本 8 FPS（+TTA），不对整片做 8 FPS 提特征 |
| 困难样本 | `refinement/tta.py` `refinement/transvcl.py` | TTA / Score Normalization / 裁边补偿（原素材中心裁切到相同视野）；TransVCL 为**可选依赖**，缺失不影响主流程 |
| 导出 | `export/` | 精确重编码（默认，CRF 16）或 stream copy；JSON / CSV / contact sheet / 对照视频 |

### 为什么必须做时序对齐

单帧相似只能说明「像」，不能说明「来自」。若成片 18.0s/18.5s/19.0s 分别匹配到素材
54.0s/12.3s/54.5s，即使单帧相似度很高，也必须**降低置信度**——因为时间轴对应关系不成立。
TN 的做法是：在候选匹配点上建有向图，边要求 query 与 source 同时递增、步长受限、
斜率落在允许的变速范围内，然后求最长加权路径；移除已用点后迭代，因此天然支持一个成片
里的**多个片段**、以及**同一素材被重复使用**。

### 采样密度为什么要逐级加密

1 FPS 粗定位与真实剪辑点最多相差 0.5s 相位。运动较快的素材上，这 0.5s 足以让**真实来源
一条时序链都建不出来**，于是被另一个素材的伪路径抢走（实测：某片段被错判到 Source02，
speed 拟合出 1.425 这种明显不合理的值）。短镜头（1~2s）在 1 FPS 下只有 1~2 个采样点，
连 `min_length` 都凑不满。

因此当粗定位的复合分（视觉 × 时序覆盖）低于 `confidence.high` 时，会依次用
`sampling.medium_fps`(2) 与 `sampling.fine_fps`(4) 对候选素材做**全片重扫**并重新对齐，
一旦足够自信立即停止；结果全部进特征缓存，重复运行不再计算。素材时长超过
`sampling.rescan_max_source_sec`(1200s) 时跳过全片重扫，避免长片代价失控。

### 变速


不要求 `Δt_query == Δt_source`，只要求 `Δt_source ≈ k·Δt_query` 且 `k` 在一个片段内稳定。
`k` 由时间域线性拟合直接给出（即报告里的 `speed`），默认允许 0.5x ~ 2.0x。

### 置信度（权重全部可配置）

```
confidence = (w_visual·visual + w_temporal·temporal + w_margin·margin + w_alignment·alignment)
             / Σw  ×  length_factor
```

- `visual`：路径上匹配点的平均余弦相似度
- `temporal`：镜头内被路径覆盖的采样帧比例
- `margin`：最佳来源与次佳来源的分差（区分度）
- `alignment`：线性拟合 R² 与残差（秒）的组合
- `length_factor`：短片段信息量不足时的衰减

分级：`HIGH / MEDIUM / LOW / UNKNOWN`。
**候选之间分数过于接近（margin < 阈值）时一律输出 UNKNOWN**，绝不为了结果好看硬给一个来源。

---

## 3. 常用参数

```bash
# 换时序对齐方法（横向对比用）
python -m source_trace -q Final.mp4 -s Sources --method hv

# 提高定位精度（更慢）
python -m source_trace -q Final.mp4 -s Sources --fine-fps 8 --hard-fps 12

# 困难样本开 TTA
python -m source_trace -q Final.mp4 -s Sources --tta

# 无 GPU / 无 torch 环境
python -m source_trace -q Final.mp4 -s Sources --device cpu
python -m source_trace -q Final.mp4 -s Sources --model colorhist

# 自定义配置
python -m source_trace -q Final.mp4 -s Sources --config my.yaml
```

`--help` 可看全部参数。所有阈值与权重都在 `source_trace/config.py`，可用 YAML/JSON 覆盖。

---

## 4. 测试数据与评测

测试数据**完全离线生成**（numpy + OpenCV 渲染 → FFmpeg 编码），不下载任何视频：

```bash
python -m source_trace.evaluation.synthetic --out testdata/multi --case multi
```

用例：

- `basic`：3 素材，纯裁剪
- `multi`：5 素材、6 片段、含短片段（1.2s）、同一素材重复使用、大量无关内容
- `robust`：缩放 / 裁边 / 重编码 / 调色 / 字幕 / 水印 / 变速 0.9x & 1.25x / 两次加工组合
- `distractor`：干扰素材（两个素材与另外两个同风格同色调）
- `unknown`：含一个来源被移除的片段，期望输出 UNKNOWN

评测与报告：

```bash
python -m source_trace.evaluation.benchmark --data testdata --out reports
python -m source_trace.evaluation.report --reports reports
```

指标：`source_recall`、`segment_recall(IoU)`、`mean_start_error`、`mean_end_error`、
`median_iou`、`median_iou_2d`、`false_positive_rate`、`unknown_accuracy`。

报告：`reports/BENCHMARK_REPORT.md`、`PERFORMANCE_REPORT.md`、`ROBUSTNESS_REPORT.md`、
`BASELINE_REPORT.md`、`FINAL_REPORT.md`、`results.json`。

单元/集成测试：

```bash
python -m pytest tests -q
```

---

## 5. 工程约定

- 时间一律 `float64` 秒，来源是 **timestamp**，不是 frame index
- 每个素材独立索引，`source_id` 全程保留
- embedding 必须记录 `model / modality / dimension / normalization`，
  不同 modality 或不同模型的向量**混用会直接抛错**（`FeatureSet.assert_compatible`）
- 单个素材损坏不影响其他素材：记入 `errors.json` 后继续
- 可选依赖（faiss / PySceneDetect / TransVCL / GUI）缺失时全部有降级路径

## 6. 目录结构

```
source_trace/
├── cli.py  config.py  pipeline.py  gui.py
├── video/       probe.py  reader.py  shot_detector.py  ffmpeg_tools.py
├── features/    isc.py  extractor.py  cache.py
├── retrieval/   index.py  search.py  candidates.py
├── temporal/    similarity.py  alignment.py  mapping.py
├── refinement/  fine_match.py  tta.py  transvcl.py
├── export/      ffmpeg.py  json_report.py  csv_report.py  visualize.py
└── evaluation/  synthetic.py  ground_truth.py  metrics.py  benchmark.py  report.py
tests/           models/           tools/ffmpeg/
```
