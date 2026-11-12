# 连续重规划：独立恢复试验准入

2026-11-12。为[独立恢复试验](../../../plans/v0.0.1/tracking/references/adaptive_live_bowl_recovery_20261112.md)
新增准入，固定独立会话和输出，核对配置、计划、历史初态、当前图像、模型响应、
原失败收据、设备和物理预算区间；不改变原试验失败状态或覆盖原始输出。
加载过程不调用历史模型适配器的全局绑定，隔离进程中旧模块全局和完整来源清单
在调用前后保持一致，也未导入 GPU 模型或仿真器。

32 并发相关回归 **65 项通过、0 skip、0 collection errors**，全部 worker 已
回收且源码前后一致；结果
`outputs/qa/adaptive_live_bowl_recovery_admission_20261112/result.json` SHA-256
`8dcdbe29c78c45a36de73cc1127a2f8836725a72c1688cfc13129b6e3cacbfc0`。
本阶段尚未接入新试验的进程监督、实际模型或物理运行；旧失败的物理消耗仍为
未知区间，最终 64 并发检查及实际预检未执行。
