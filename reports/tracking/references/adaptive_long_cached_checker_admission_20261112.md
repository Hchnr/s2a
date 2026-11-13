# 长窗失败后的缓存 checker 准入

2026-11-12。为[独立缓存 checker](../../../plans/v0.0.1/tracking/references/adaptive_long_cached_checker_20261112.md)
新增准入：重新运行只读部分审计，核对原生产者、失败 checker、实际 worker
退出及此前累计区间；新输出保留原模型会话身份并隔离事件图像目录。

实际历史收据可在 CPU 环境加载，未导入模型或仿真器。测试覆盖失败来源、
累计成本、物理与模型零工作预算、会话和输出路径变化，以及部分审计被篡改
时的拒绝。32 并发相关回归 **74 项通过、0 skip、0 collection errors**，
所有 worker 实际 wait/reap，源码前后一致；结果
`outputs/qa/adaptive_long_cached_checker_admission_20261112/result.json` SHA-256
`3b5360d1de28fc1642911ca7963c0fe36ade3e5f6344c70d895e9c19bb355f7b`。

本阶段没有新增模型请求或物理步；缓存 checker 接线、最终回归与独立重放
仍待完成。原失败和任务成功的单端点中断均保持原记录。
