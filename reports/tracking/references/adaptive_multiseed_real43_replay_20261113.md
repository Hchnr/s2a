# 真实 seed 43 的保存图像重放客户端

2026-11-13。按[预登记](../../../plans/v0.0.1/tracking/references/adaptive_multiseed_real43_20261113.md)
新增单帧离线客户端：使用已核对的保存图像与历史端点身份建立一次 seed 43
请求，接收后校验 seed、图像、模型、35 去噪、mask 和数组，再与旧 seed 42
数组比较。记录中的端点并非本次现场状态；客户端不创建仿真模型、推进物理
或派发 action。

输入图像在请求前后均核对 SHA；错误 seed、图像、去噪数、mask、无效槽和
非有限值均在合成测试中拒绝。32 并发相关回归 **100 项通过、0 skip、0
collection errors**，源码前后一致，worker 全部实际 wait/reap；结果
`outputs/qa/adaptive_multiseed_real43_replay_20261113/result.json` SHA-256
`0c4261c5307d40e6f1b0d088f7b83e6c97d32c05f48714e1e24fd74fa8be2a2d`。
本阶段没有真实模型初始化、GPU 推理、物理步或最终留出运行；运行监督和
真实模型验证仍待完成。
