# 连续重规划：隔离仿真 worker

2026-11-12。新增独立生产与 checker 仿真 worker，直接组合已核验的 reset 和
历史参考路径，不修改历史运行模块；在固定 115 周期预算内接入变长生产或重放。
checker 在重放前核对每个保存响应与实际模型 worker 输出及当前图像的身份，
并检查候选策略未加载。失败时保留可取得的部分 runtime 收据。

32 并发相关回归 **51 项通过、0 skip、0 collection errors**；包含历史输入
组合、响应及去噪计数篡改、错误登记和 checker 导入隔离。检查记录
`outputs/qa/adaptive_live_bowl_worker_20261112/result.json` SHA-256
`ec08da5dfd22d0e824e916326b889b0c36d217e2180e35e938ad13a0d70e14e8`。
外层进程所有者和预检尚未完成；本阶段无新真实模型、GPU、物理仿真或留出试验。
