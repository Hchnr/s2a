# 连续重规划：真实模型 worker 准入

2026-11-12。新增有界模型 worker：加载权重前核对显式 GPU UUID、空 LD、
单线程环境、当前图像目录、任务、seed、去噪和总周期预算；通过模型来源与
OpenCV 准入后才初始化一次模型，并以原主动槽掩码和独立计数器服务当前图像。
worker 保存实际尝试数和部分失败，外层仍须核对进程 wait/reap 与 GPU 释放。

32 并发相关回归 **59 项通过、0 skip、0 collection errors**，包括不匹配的
设备掩码和非法登记在模型初始化前失败；CPU 测试的 CUDA 均为空。检查记录
`outputs/qa/adaptive_model_worker_20261112/result.json` SHA-256
`79fc37ee0277495f9d2588207c15b88c55064e6774a13900532d2ccbd0ab8b29`。
真实进程所有者和预检仍待接入；本阶段无新真实模型、GPU、物理仿真或留出试验。
