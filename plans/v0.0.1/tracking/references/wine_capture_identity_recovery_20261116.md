# wine step104 身份诊断的受控目录修复验证

2026-11-16。此前单独登记的双路径诊断在两条各 104 周期之后，因 capture
元数据写到受控 `runtime/` 目录之外而失败，见[事件报告](../../../../reports/tracking/references/wine_capture_identity_incident_20261116.md)。
本次将输出目标改为各 worker 自己的 `runtime/capture104.json`，维持原目录
保护和原 capture 身份比较；先用有写入限制的子进程测试实际持久化。

另登记一次 producer 重算和一次 checker 原动作重放，均以相同 seed42、旧
wine 初态及 step104 为界。两条仅使用 step96 及此前已生成参考，逐周期核对
原动作和状态；在渲染 step104 后保存原 / 实际 capture，再逐字段运行原比较器，
不解码 step104 参考、不派发该步动作。来源资格核验可以离线检查完整原始记录。
不修改、覆盖或追认为此前六例对照及诊断失败的通过；差异未复现时保持不确定。

本次新增预算为每条最多 104 控制周期 / 2,600 内步、3 个模型及 3 个 OSMesa
context，合计最多 208 周期 / 5,200 内步、6 模型 / context、2 个 CPU worker；
此前失败诊断另已消耗 208 / 5,200、6 模型 / context，累计上限为 416 / 10,400。
这不包含更早六例对照中 wine / seed42 的 104 / 2,600；该旧开销仍独立保留。
使用独占新输出、先跑受控路径测试及来源绑定的回归 / 预检，所有 worker 实际
wait/reap，不自动重试。新 WAM、GPU 和 fresh 初态均为 0。
