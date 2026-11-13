# 碗任务原生谓词静态诊断准备

2026-11-13。[诊断计划](../../../plans/v0.0.1/tracking/references/bowl_native_predicate_probe_20261113.md)
已接入原生产者与完整独立重放的 40 个自适应保存端点，并绑定原 XML、资产、
BDDL、原生 `On` 和碰撞分组代码。静态查询记录高度、水平距离及碗盘碰撞 geom
接触，逐端点与已保存任务标签比较；不一致时明确拒绝把重建当成全部标签的解释。

测试覆盖 0.03 m 的严格边界、碰撞 geom 分组、标签不一致，以及在静态 forward
中禁止 `mj_step` 并恢复原函数；实际来源准入验证了 40/10 个碗/盘 collision
geom 和第 129 端点的已有失败标签。32 并发相关回归 **61 项通过、0 skip、
0 collection errors**，源码前后一致、所有 worker 实际 wait/reap；结果
`outputs/qa/bowl_predicate_development_20261113/result.json` SHA-256
`3e949658cee1cb294a9889faaec071446b5b7338a5e7c82104cb4862e5c1cc6a`。

更正：上述 32 并发回归包含一个真实 CPU MuJoCo 合成测试，创建 1 个小模型/
data 并执行 1 次非积分 forward；原报告误称该阶段零模型/forward。没有新
物理步、渲染 context 或模型请求，实际保存状态的静态查询仍未执行。测试开销
与后续检查路径修订见[计划](../../../plans/v0.0.1/tracking/references/bowl_native_predicate_probe_20261113.md)。
上述合成及准入测试不解释第 129 端点的物理原因，也不构成稳定任务资格。
