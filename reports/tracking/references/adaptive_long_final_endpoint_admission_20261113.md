# 最终端点 checker 的历史准入

2026-11-13。[新预登记](../../../plans/v0.0.1/tracking/references/adaptive_long_final_endpoint_checker_20261113.md)
的准入重新核对原生产者、原第 147 端点失败、缓存 checker 第 119 端点失败、
只读事件审计及两次失败后的累计开销。新输出沿用原模型会话、seed 和控制条件，
只允许在第 147 端点启用缓存，不导入模型或仿真器。

配置与历史输入篡改、缓存步号变化、自动重试、模型请求以及旧输出路径均会被
拒绝。32 并发相关回归 **55 项通过、0 skip、0 collection errors**，所有
worker 实际 wait/reap、源码前后一致；结果
`outputs/qa/final_endpoint_admission_20261113/result.json` SHA-256
`8a7193b305a47cd7cfacb786d939bb3ea7093e40a415fc4f3b6923dd1d145f7f`。

本阶段没有新增模型请求或物理步；最终端点 worker、监督及独立重放仍待完成。
原任务第 129 端点失败和末尾仅 18 周期成功均未改变。
