# 连续重规划：40 周期试验准入

2026-11-12。为[长窗开发试验](../../../plans/v0.0.1/tracking/references/adaptive_live_bowl_long_20261112.md)
增加独立登记准入：重新核验上一轮生产者、模型、checker、worker 回收和独立
物理重放，按实际收据累加开销并保留更早失败的未知物理区间。新会话、输出、
40 周期及 40 请求上限和固定身份均独立绑定，不修改历史源码或试验产物。

真实历史输入在 CPU Python 与登记的模型 Python 中得到相同输入收据，未加载
模型或仿真器。测试覆盖旧 checker 标签、累计区间、来源登记或实际回收状态被
篡改的拒绝，以及新配置的预算、类型、身份和输出路径边界。

32 并发相关回归 **91 项通过、0 skip、0 collection errors**，全部 worker
实际 wait/reap，源码前后一致。结果
`outputs/qa/adaptive_live_bowl_long_admission_20261112/result.json` SHA-256
`2ae575ab4366848f0ee4cb3cc272214266cb0f2780850db0e8a0c9c1144fd4c8`；
新增文件 Ruff 检查和格式检查通过。

本阶段没有新增模型请求或物理步；可复用监督器、最终 64 并发检查、预检和
真实长窗试验仍待完成。当前模型 worker 仍固定 seed 42，不能覆盖最终验收的
seed 43；多任务及 600 周期、留出验收仍未完成。
