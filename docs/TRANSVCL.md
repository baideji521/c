# TransVCL 可选集成说明

TransVCL 在本项目中的定位是 **困难样本判别器（Difficult Sample Resolver）**，
不是第一层算法。只有当 `ISC + VCSL/TN` 给出的置信度低于
`confidence.enhance_threshold` 且候选之间存在歧义时才会被调用。

## 为什么做成可选依赖

- 官方实现依赖较重（自定义 CUDA 环境 / 额外权重），在部分环境难以安装
- 本项目的硬性要求是「没有 TransVCL 主程序依然能跑」

因此 `source_trace/refinement/transvcl.py` 只暴露三个函数：

- `is_available()`：同时检查 `transvcl` 包与 `models/transvcl/*.pth` 权重
- `unavailable_reason()`：说明缺什么
- `resolve(...)`：不可用时返回 `None`，调用方保留原结果

benchmark 组合 `F` 在检测到不可用时，会记录
`FAILURE_REASON` 并跳过该组合，而不是让整个评测失败。

## 启用步骤

1. 安装依赖（参考上游仓库 <https://github.com/transvcl/TransVCL> 的 README）
2. 下载权重放入 `models/transvcl/`，例如 `models/transvcl/transvcl_model.pth`
3. 在 `refinement/transvcl.py` 的 `resolve()` 中实现推理适配层：
   输入为「query 片段特征 + 候选素材特征 + 粗定位区间」，
   输出为「片段级置信度 / 修正后的起止时间」
4. 运行时加 `--config` 打开 `refinement.transvcl: true`

当前状态：**未安装**，主流程不受影响。
