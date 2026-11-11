# step125 质量搜索的缓存约束审核

2026-11-11，按[登记](../../../plans/v0.0.1/tracking/joint_control/cached_live_stop125_constraint_audit_20261111.md)，
针对[前次静态诊断](cached_live_stop125_feasibility_20261111.md)的 `quality_past`
分支，在相同已保存状态、参考、过去误差和移位种子上重新运行一次原求解器，并在
隔离命名空间读取它已经计算的候选缓存；没有更改目标、约束、容差或候选选择。
本轮与前轮保存的当前误差、种子、误差历史、优化状态、迭代数和 58 次响应数一致；
前轮未保存完整候选缓存，不能声称逐候选重现。更早真实 WAM 停止时的原种子也
未保存，此处不是原现场搜索的精确复现。

58 个缓存候选中，54 个未通过绝对位置门、1 个未通过旋转门、26 个未通过
动作变化率门，失败项可以重叠；没有候选通过全部原后置检查。其中 **4 个只因
变化率被拒绝**。优化器最终候选的预测结果如下：

| 项目 | 结果 | 原后置检查 |
| --- | ---: | --- |
| 位置误差 | 0.010000000000006 m | ≤ 0.01 m + `1e-10` m，通过 |
| 旋转误差 | 0.03213206850 rad | ≤ 0.05 rad + `1e-10` rad，通过 |
| 最大动作变化 | 0.250000000006851 | ≤ 0.25 + `1e-12`，不通过 |
| 最小 box 余量 | `2.43e-12` | 通过 |
| 最小关节/制动收紧余量 | 0.715421 rad | 通过 |

最终候选的动作变化比名义 0.25 高约 `6.85e-12`，比既有允许容差上界高约
`5.85e-12`；4 个仅变化率失败的候选中，最接近者比名义限制高约 `4.14e-12`。
该分支没有终端方向约束。优化器报告成功终止与严格后置过滤不矛盾：它的最终点
位于动作变化率的数值边界之外。以上只解释**本次复核分支**的现象；另两个质量
分支未做缓存审核，也未证明真实 WAM 原停止由同一因素造成。

一次模型初始化、一次静态 forward、一次优化、58 次局部模型响应；缓存审核
额外模型响应为 0，未执行物理积分、新 WAM/GPU 调用、新初态或独立 witness
核验。工作进程正常退出并实际 wait/reap；运行前 32 并发相关专项 37 项通过、
无跳过或收集错误。所有候选均未派发，修正候选后仍须遵守原阈值并经独立复算；
静态结果不授予物理、稳定成功或留出资格。

| 仓库内证据 | SHA256 |
| --- | --- |
| [零工作 dry](../../../outputs/cached_live_stop125_constraint_audit_dry_20261111/result.json) | `4835d234b9d6bd0d0e6c49edfb1e19a358ba9b82a29c2e9e2a57d7ec23cb9174` |
| [32 并发专项](../../../outputs/qa/cached_live_stop125_constraint_audit_20261111/result.json) | `abdcab8a387ffe724a93ed03cba487dfbf330674bd82af13fe28de93dd962d76` |
| [运行登记](../../../outputs/cached_live_stop125_constraint_audit_20261111/registration.json) | `4a927f3da0d3262865b0321a86f38d8cb142e1faecbd66a2f8b79f5d00128473` |
| [owner 结果](../../../outputs/cached_live_stop125_constraint_audit_20261111/result.json) | `5cafa4df1990471afa90e1f37d83bbc5f410ed679111c45e782bf7b92b91e4a8` |
| [全部缓存及过滤结果](../../../outputs/cached_live_stop125_constraint_audit_20261111/worker/result.json) | `30625ec96570a85113597b9122ad25eb4888a81feb7117578660a4315080030b` |
| [进程收尾](../../../outputs/cached_live_stop125_constraint_audit_20261111/workers.json) | `52fc86e2e2a218b007fa0d41abc4f4390cfe6fa26dfb5b49b3d7ac1665df41e3` |
