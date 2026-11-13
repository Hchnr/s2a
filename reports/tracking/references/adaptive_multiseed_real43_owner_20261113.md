# 真实 seed 43 重放的进程监督准备

2026-11-13。单次保存图像重放已接入独立 CPU 客户端、单 UUID 模型 worker、
受限 Unix 帧通道和实际 wait/reap 的双进程监督。监督器独立核对模型实际请求、
响应、去噪与 mask 收据，以及客户端接收的数组；历史端点仅作为来源身份。
正常完整收据记录精确开销，worker 未回收、来源不匹配或收据缺失时保留明确
上下界；不自动重试。

CPU 合成测试覆盖正常流程、响应身份篡改、mask/去噪收据异常及 worker 失败
时的未知成本范围；实际模型 Python 的空 CUDA 导入检查确认入口不会提前导入
OpenCV、Torch 或 Cosmos。32 并发相关回归 **112 项通过、0 skip、0 collection
errors**，源码前后一致，全部测试子进程实际 wait/reap；结果
`outputs/qa/adaptive_multiseed_real43_owner_v2_20261113/result.json` SHA-256
`eef1a0c8c8c90bbcd7cb7f954a9a46fe7788f799202f42cba7e155444076d30e`。

本阶段未初始化真实模型、使用 GPU、执行物理或访问最终留出初态；仍须完成
64 并发准入、dry、实际权重/源码哈希、OpenCV 探针和 GPU 空闲核验后才能运行。
