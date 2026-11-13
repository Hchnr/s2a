# 当前图像模型交换的独立核验

2026-11-13。监督层在放行独立 checker 前，重新读取真实模型请求、响应、
当前图像绑定和通道信封，核对 seed、任务、checkpoint、有效 mask、35 次
去噪证据及原始 action / normalized action；无请求时只接受已记录的正确
停止帧，不虚构采样。

已用历史 task2 / seed42 的真实模型请求、响应和物理 producer 收据验证
完整链路；旧 worker 未显式存储的完成标记仅在测试内补充，不修改历史
产物。合成篡改 seed、响应派发标记、去噪记录和图像身份均被拒绝。
32 并发相关 CPU 回归 **81 项通过、0 skip、0 collection errors**，worker
均 wait/reap、来源前后一致。结果
`outputs/qa/reset_seed43_model_exchange_20261113/result.json` SHA-256
`c749dc5f78a2790323437146ebc8b6422f9802f34838356e704644456bd774f9`。

本阶段没有新模型初始化、WAM 请求、物理步或最终留出案例。历史校验
只证明审计工具能核对已保存记录，不是新的 seed43 闭环或任务成功。
