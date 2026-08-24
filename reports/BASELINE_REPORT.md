# BASELINE_REPORT

## 环境

- os: Windows 10
- python: 3.11.9
- cpu: Intel64 Family 6 Model 79 Stepping 1, GenuineIntel
- torch: 2.6.0+cu124
- cuda_available: True
- gpu: NVIDIA GeForce RTX 3060
- vram_mb: 12287
- cuda_version: 12.4
- ffmpeg: ffmpeg version 9.0.1-essentials_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers
- model: isc21

## 基线（组合 A：ISC only，无时序约束、单候选）

- 用例 basic：source_recall=1.0000 segment_recall=0.3333 start_err=5.830 s IoU=0.4978 FPR=0.0000
- 用例 multi：source_recall=1.0000 segment_recall=0.3333 start_err=9.438 s IoU=0.0385 FPR=0.0000
- 用例 robust：source_recall=1.0000 segment_recall=0.3000 start_err=18.925 s IoU=0.1663 FPR=0.0000
- 用例 distractor：source_recall=1.0000 segment_recall=1.0000 start_err=0.000 s IoU=1.0000 FPR=0.0000
- 用例 unknown：source_recall=1.0000 segment_recall=1.0000 start_err=0.000 s IoU=0.7500 FPR=0.3333

## 与推荐组合（D）对比

- A: source_recall=1.0000 segment_recall=0.5933 start_err=6.839 s IoU=0.4905 FPR=0.0667
- B: source_recall=0.9600 segment_recall=0.5933 start_err=6.599 s IoU=0.5070 FPR=0.0000
- C: source_recall=0.5600 segment_recall=0.5600 start_err=0.051 s IoU=0.9850 FPR=0.3000
- D: source_recall=1.0000 segment_recall=1.0000 start_err=0.039 s IoU=0.9778 FPR=0.0000
