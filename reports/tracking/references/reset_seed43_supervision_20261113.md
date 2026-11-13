# 独立 checker 的 producer 前置核验

2026-11-13。checker 派发前，重新读取并核验 producer 的登记、模型 bundle、
两个进程的实际 wait/reap 和日志、GPU 释放状态、物理结果、当前图像模型
交换及成本上下界。模型或物理收据、计数、来源、日志和 GPU 状态任一不符
时，不启动 checker。

测试使用历史真实模型和物理收据构造封闭 producer；核对 16 周期 / 400 内步、
一次模型请求及 35 次去噪，并验证来源、日志、进程回收、成本与 GPU 状态
篡改均被拒绝。32 并发相关 CPU 回归 **66 项通过、0 skip、0 collection
errors**，worker 均 wait/reap、来源前后一致。结果
`outputs/qa/reset_seed43_supervision_20261113/result.json` SHA-256
`90692e611766881e955702f6c3cd8719b94860ca83cf7ccf29cc9870f809773b`。

此阶段没有新模型调用、物理步或留出案例；实际父进程监督和 seed43
两窗口试验尚未执行。
