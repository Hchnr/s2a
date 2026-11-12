# 碗任务保存状态几何诊断：状态逐位检查失败

2026-11-12。按[原计划](../../../plans/v0.0.1/tracking/references/reset_bowl_saved_geometry_20261112.md)
构造一个 CPU MuJoCo 模型并读取保存状态。第 10 次 `mj_forward` 后，状态逐位不变
检查失败，进程终止，没有发布 TCP–碗距离结论。原 worker 返回
`A non-stepping geometry query must not change the saved state`；这只说明检查发现
变化，原记录未保存变更分量和幅度，**尚不能确定变化来自四元数归一化**。

实际消耗为一个 MuJoCo 模型和 data、10 次已完成 forward、零物理步、零渲染 context、
零世界模型调用；worker 退出码 1、未收到信号且已实际 wait/reap。原结果
`outputs/reset_bowl_saved_geometry_20261112/result.json` SHA-256
`18ea6ee0293f0dcb627eacfefb3673699be21fc9b0623b0ce09367836faaec57`，worker 结果
SHA-256 `784980f2dc41d5e9f1fb304b6cfa58aff3e25c06236df6abb77a18a653e05a9c`。
本次不自动重试，不覆盖失败结果；若另作新诊断，必须预先登记状态变化允许范围、
实际差值、输入身份及累计预算，非允许的变化继续拒绝。
