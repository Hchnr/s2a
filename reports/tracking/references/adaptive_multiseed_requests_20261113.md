# 双 seed 自适应请求循环

2026-11-13。[预登记](../../../plans/v0.0.1/tracking/references/adaptive_multiseed_interface_20261113.md)
的独立请求循环将登记 seed 42/43 用于每次 Python、NumPy、Torch 和 CUDA
随机数重置，并核对当前图像、任务、请求、实际响应及发送 envelope 的身份。
无效 seed 或篡改的请求在模型准备前拒绝；响应身份不符、图像变化和部分去噪
失败均不发送参考，已完成的工作仍记账。旧真实实验使用的文件没有修改。

CPU 合成服务覆盖三任务乘两个 seed、可变请求间隔、提前停止和失败边界；
seed 42 的三次请求（当前图像路径除外）和响应 envelope 与旧循环一致。
32 并发相关回归 **82 项通过、0 skip、0 collection errors**，源码前后一致、
子进程全部实际 wait/reap；结果
`outputs/qa/adaptive_multiseed_requests_20261113/result.json` SHA-256
`3b0e7bb62874697e209fa480857273089b1dbbdd0a3cdb98702536446420e84e`。

首次直接 smoke 的 pytest 临时目录父目录未创建，18 个测试在 fixture 准备时
报 `FileNotFoundError`，没有执行测试逻辑；创建独立目录后重跑 18 项通过。
合成服务不构成真实模型采样。本阶段没有新增模型初始化、WAM 请求、物理步
或最终留出运行；独立模型 worker 和真实 seed 43 验证仍待完成。
