# 双 seed 模型请求合同

2026-11-13。[预登记](../../../plans/v0.0.1/tracking/references/adaptive_multiseed_interface_20261113.md)
的基础合同已接收整数 seed 42/43，并将登记值用于请求名称、模型请求、响应
身份和 Python/NumPy/Torch/CUDA 随机数重置；三个任务仍共用原相机、H40、
20 Hz、35 去噪和 mask 合同。seed 42 请求在三任务上与旧构造逐字段一致，
seed、任务、响应、相机或采样身份不匹配均拒绝。

32 并发相关回归 **89 项通过、0 skip、0 collection errors**，源码前后一致，
所有 worker 实际 wait/reap；结果
`outputs/qa/adaptive_multiseed_contract_20261113/result.json` SHA-256
`e22f98cacda4b95fcccdde7674ee64ca66fb783c5507980abaeb3897a4a3ba59`。
本阶段仅测试独立合同和合成随机数调用，未接入自适应服务循环或真实模型，
没有新增模型初始化、WAM 请求、物理步或最终留出运行。
