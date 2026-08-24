# BENCHMARK_REPORT

## 运行环境

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

## 各算法组合 × 用例 指标

### A / basic
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 0.3333
- mean_start_error: 5.8300 s
- mean_end_error: 5.6700 s
- median_iou(source): 0.4978
- median_iou_2d: 0.4978
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 3，耗时: 7.82 s

### A / multi
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 0.3333
- mean_start_error: 9.4378 s
- mean_end_error: 9.6717 s
- median_iou(source): 0.0385
- median_iou_2d: 0.0385
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 6，耗时: 12.05 s

### A / robust
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 0.3000
- mean_start_error: 18.9247 s
- mean_end_error: 18.0920 s
- median_iou(source): 0.1663
- median_iou_2d: 0.1663
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 10，耗时: 6.15 s

### A / distractor
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.0000 s
- mean_end_error: 0.0000 s
- median_iou(source): 1.0000
- median_iou_2d: 1.0000
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 4，耗时: 9.67 s

### A / unknown
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.0000 s
- mean_end_error: 2.0000 s
- median_iou(source): 0.7500
- median_iou_2d: 0.7500
- false_positive_rate: 0.3333
- unknown_accuracy: 0.0000
- 片段数: 3，耗时: 3.69 s

### B / basic
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 0.3333
- mean_start_error: 5.8300 s
- mean_end_error: 5.6700 s
- median_iou(source): 0.4978
- median_iou_2d: 0.4978
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 3，耗时: 3.11 s

### B / multi
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 0.3333
- mean_start_error: 9.4378 s
- mean_end_error: 9.6717 s
- median_iou(source): 0.0385
- median_iou_2d: 0.0385
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 6，耗时: 3.82 s

### B / robust
- source_recall: 0.8000
- segment_recall(IoU>=0.50): 0.3000
- mean_start_error: 17.7259 s
- mean_end_error: 16.3975 s
- median_iou(source): 0.2489
- median_iou_2d: 0.2489
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 10，耗时: 5.04 s

### B / distractor
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.0000 s
- mean_end_error: 0.0000 s
- median_iou(source): 1.0000
- median_iou_2d: 1.0000
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 4，耗时: 3.68 s

### B / unknown
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.0000 s
- mean_end_error: 2.0000 s
- median_iou(source): 0.7500
- median_iou_2d: 0.7500
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 3，耗时: 2.99 s

### C / basic
- source_recall: 0.3333
- segment_recall(IoU>=0.50): 0.3333
- mean_start_error: 0.0000 s
- mean_end_error: 0.0000 s
- median_iou(source): 1.0000
- median_iou_2d: 1.0000
- false_positive_rate: 0.5000
- unknown_accuracy: 1.0000
- 片段数: 3，耗时: 2.97 s

### C / multi
- source_recall: 0.1667
- segment_recall(IoU>=0.50): 0.1667
- mean_start_error: 0.1200 s
- mean_end_error: 0.1200 s
- median_iou(source): 0.9469
- median_iou_2d: 0.9469
- false_positive_rate: 0.5000
- unknown_accuracy: 1.0000
- 片段数: 6，耗时: 3.58 s

### C / robust
- source_recall: 0.3000
- segment_recall(IoU>=0.50): 0.3000
- mean_start_error: 0.1333 s
- mean_end_error: 0.1467 s
- median_iou(source): 0.9781
- median_iou_2d: 0.9781
- false_positive_rate: 0.5000
- unknown_accuracy: 1.0000
- 片段数: 10，耗时: 5.08 s

### C / distractor
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.0000 s
- mean_end_error: 0.0000 s
- median_iou(source): 1.0000
- median_iou_2d: 1.0000
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 4，耗时: 3.42 s

### C / unknown
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.0000 s
- mean_end_error: 0.0000 s
- median_iou(source): 1.0000
- median_iou_2d: 1.0000
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 3，耗时: 2.94 s

### C-dp / basic
- source_recall: 0.3333
- segment_recall(IoU>=0.50): 0.3333
- mean_start_error: 0.0000 s
- mean_end_error: 0.0000 s
- median_iou(source): 1.0000
- median_iou_2d: 1.0000
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 3，耗时: 3.09 s

### C-dp / multi
- source_recall: 0.1667
- segment_recall(IoU>=0.50): 0.1667
- mean_start_error: 0.1200 s
- mean_end_error: 0.1200 s
- median_iou(source): 0.9469
- median_iou_2d: 0.9469
- false_positive_rate: 0.5000
- unknown_accuracy: 1.0000
- 片段数: 6，耗时: 3.74 s

### C-dp / robust
- source_recall: 0.3000
- segment_recall(IoU>=0.50): 0.3000
- mean_start_error: 0.1333 s
- mean_end_error: 0.1467 s
- median_iou(source): 0.9781
- median_iou_2d: 0.9781
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 10，耗时: 4.66 s

### C-dp / distractor
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.0000 s
- mean_end_error: 0.0000 s
- median_iou(source): 1.0000
- median_iou_2d: 1.0000
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 4，耗时: 3.50 s

### C-dp / unknown
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.0000 s
- mean_end_error: 0.0000 s
- median_iou(source): 1.0000
- median_iou_2d: 1.0000
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 3，耗时: 3.10 s

### C-dtw / basic
- source_recall: 0.3333
- segment_recall(IoU>=0.50): 0.3333
- mean_start_error: 0.0000 s
- mean_end_error: 0.0000 s
- median_iou(source): 1.0000
- median_iou_2d: 1.0000
- false_positive_rate: 0.6667
- unknown_accuracy: 1.0000
- 片段数: 3，耗时: 2.98 s

### C-dtw / multi
- source_recall: 0.1667
- segment_recall(IoU>=0.50): 0.1667
- mean_start_error: 0.1200 s
- mean_end_error: 0.1200 s
- median_iou(source): 0.9469
- median_iou_2d: 0.9469
- false_positive_rate: 0.6667
- unknown_accuracy: 1.0000
- 片段数: 6，耗时: 3.79 s

### C-dtw / robust
- source_recall: 0.3000
- segment_recall(IoU>=0.50): 0.3000
- mean_start_error: 0.1333 s
- mean_end_error: 0.1467 s
- median_iou(source): 0.9781
- median_iou_2d: 0.9781
- false_positive_rate: 0.6667
- unknown_accuracy: 1.0000
- 片段数: 10，耗时: 4.82 s

### C-dtw / distractor
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.1250 s
- mean_end_error: 0.3750 s
- median_iou(source): 1.0000
- median_iou_2d: 1.0000
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 4，耗时: 3.48 s

### C-dtw / unknown
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.0000 s
- mean_end_error: 0.0000 s
- median_iou(source): 1.0000
- median_iou_2d: 1.0000
- false_positive_rate: 0.3333
- unknown_accuracy: 0.0000
- 片段数: 3，耗时: 3.00 s

### C-hv / basic
- source_recall: 0.6667
- segment_recall(IoU>=0.50): 0.3333
- mean_start_error: 0.0000 s
- mean_end_error: 1.1300 s
- median_iou(source): 0.7489
- median_iou_2d: 0.7489
- false_positive_rate: 0.3333
- unknown_accuracy: 1.0000
- 片段数: 3，耗时: 3.00 s

### C-hv / multi
- source_recall: 0.1667
- segment_recall(IoU>=0.50): 0.1667
- mean_start_error: 0.1200 s
- mean_end_error: 0.1200 s
- median_iou(source): 0.9469
- median_iou_2d: 0.9469
- false_positive_rate: 0.5000
- unknown_accuracy: 1.0000
- 片段数: 6，耗时: 3.61 s

### C-hv / robust
- source_recall: 0.5000
- segment_recall(IoU>=0.50): 0.4000
- mean_start_error: 0.7220 s
- mean_end_error: 0.5240 s
- median_iou(source): 0.8657
- median_iou_2d: 0.8657
- false_positive_rate: 0.4444
- unknown_accuracy: 1.0000
- 片段数: 10，耗时: 4.96 s

### C-hv / distractor
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.0000 s
- mean_end_error: 0.0000 s
- median_iou(source): 1.0000
- median_iou_2d: 1.0000
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 4，耗时: 3.87 s

### C-hv / unknown
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.0000 s
- mean_end_error: 0.0000 s
- median_iou(source): 1.0000
- median_iou_2d: 1.0000
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 3，耗时: 2.93 s

### D / basic
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.0067 s
- mean_end_error: 0.0067 s
- median_iou(source): 0.9956
- median_iou_2d: 0.9956
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 3，耗时: 6.79 s

### D / multi
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.0620 s
- mean_end_error: 0.0995 s
- median_iou(source): 0.9541
- median_iou_2d: 0.9541
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 6，耗时: 45.52 s

### D / robust
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.1264 s
- mean_end_error: 0.2569 s
- median_iou(source): 0.9393
- median_iou_2d: 0.9393
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 10，耗时: 59.09 s

### D / distractor
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.0000 s
- mean_end_error: 0.0000 s
- median_iou(source): 1.0000
- median_iou_2d: 1.0000
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 4，耗时: 8.27 s

### D / unknown
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.0000 s
- mean_end_error: 0.0000 s
- median_iou(source): 1.0000
- median_iou_2d: 1.0000
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 3，耗时: 30.77 s

### E / basic
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.0067 s
- mean_end_error: 0.0067 s
- median_iou(source): 0.9956
- median_iou_2d: 0.9956
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 3，耗时: 6.26 s

### E / multi
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.0678 s
- mean_end_error: 0.1027 s
- median_iou(source): 0.9541
- median_iou_2d: 0.9541
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 6，耗时: 18.22 s

### E / robust
- source_recall: 0.9000
- segment_recall(IoU>=0.50): 0.9000
- mean_start_error: 0.1101 s
- mean_end_error: 0.3168 s
- median_iou(source): 0.9440
- median_iou_2d: 0.9440
- false_positive_rate: 0.1000
- unknown_accuracy: 1.0000
- 片段数: 10，耗时: 37.05 s

### E / distractor
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.0000 s
- mean_end_error: 0.0000 s
- median_iou(source): 1.0000
- median_iou_2d: 1.0000
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 4，耗时: 7.97 s

### E / unknown
- source_recall: 1.0000
- segment_recall(IoU>=0.50): 1.0000
- mean_start_error: 0.0000 s
- mean_end_error: 0.0000 s
- median_iou(source): 1.0000
- median_iou_2d: 1.0000
- false_positive_rate: 0.0000
- unknown_accuracy: 1.0000
- 片段数: 3，耗时: 11.64 s

### F / basic：失败
- FAILURE_REASON: TransVCL 不可用（未安装可选依赖或缺少权重），已跳过该组合

### F / multi：失败
- FAILURE_REASON: TransVCL 不可用（未安装可选依赖或缺少权重），已跳过该组合

### F / robust：失败
- FAILURE_REASON: TransVCL 不可用（未安装可选依赖或缺少权重），已跳过该组合

### F / distractor：失败
- FAILURE_REASON: TransVCL 不可用（未安装可选依赖或缺少权重），已跳过该组合

### F / unknown：失败
- FAILURE_REASON: TransVCL 不可用（未安装可选依赖或缺少权重），已跳过该组合

## 组合汇总（跨用例平均）

- **A**：source_recall=1.0000  segment_recall=0.5933  start_err=6.8385s  end_err=7.0867s  IoU=0.4905  FPR=0.0667  耗时=7.88s  峰值显存=687.8MB
- **B**：source_recall=0.9600  segment_recall=0.5933  start_err=6.5987s  end_err=6.7478s  IoU=0.5070  FPR=0.0000  耗时=3.73s  峰值显存=687.8MB
- **C**：source_recall=0.5600  segment_recall=0.5600  start_err=0.0507s  end_err=0.0533s  IoU=0.9850  FPR=0.3000  耗时=3.60s  峰值显存=687.8MB
- **C-dp**：source_recall=0.5600  segment_recall=0.5600  start_err=0.0507s  end_err=0.0533s  IoU=0.9850  FPR=0.1000  耗时=3.62s  峰值显存=687.8MB
- **C-dtw**：source_recall=0.5600  segment_recall=0.5600  start_err=0.0757s  end_err=0.1283s  IoU=0.9850  FPR=0.4667  耗时=3.61s  峰值显存=687.8MB
- **C-hv**：source_recall=0.6667  segment_recall=0.5800  start_err=0.1684s  end_err=0.3548s  IoU=0.9123  FPR=0.2556  耗时=3.67s  峰值显存=687.8MB
- **D**：source_recall=1.0000  segment_recall=1.0000  start_err=0.0390s  end_err=0.0726s  IoU=0.9778  FPR=0.0000  耗时=30.09s  峰值显存=687.8MB
- **E**：source_recall=0.9800  segment_recall=0.9800  start_err=0.0369s  end_err=0.0852s  IoU=0.9787  FPR=0.0200  耗时=16.23s  峰值显存=788.4MB

## 结论

- 综合最优：**D**（segment_recall=1.0000）
- 最快：**C**（平均 3.60s，segment_recall=0.5600）
- 时间定位最准：**C**（起止平均误差 0.0520s，segment_recall=0.5600）
- 误匹配最低：**B**（FPR=0.0000，segment_recall=0.5933）
- 峰值显存最高：**E**

> 注意：start_err / end_err / IoU 只在「已正确匹配到来源」的片段上统计，因此 recall 低的组合这些数值会偏乐观，必须与 segment_recall 一起看。
