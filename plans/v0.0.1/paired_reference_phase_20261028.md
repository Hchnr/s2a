# 同总时长的提前/临近参考相位分配（候选计划，未执行）

总目标及原全部门槛保持。本文只登记下一项 CPU 机制对照，入口/测试/checker 尚未实现，本轮只提交计划。GPU 继续禁用，不启动搜索、模型或物理，不修改已执行源码。

## 动机与必须先关闭的证据

原 H12 全窗口搜索及完整独立复算没有完整 witness。本轮 prefix 的 4000 响应/54 迭代将连续预测达标点 6→9；cursor 9 仍为 0.01151628224m/0.0575804846rad，最紧制动项在 cursor 6 第五关节，非唯一根因证明。其完整独立复算尚未关闭，见 [当前报告](../../reports/prefix_preserving_trajectory_control_20261028.md:31)。

执行前须同时满足：

1. prefix checker 全部 4000 响应、来源与 owner 实际 wait/reap 关闭；失败先处理资格问题。
2. [保存轨迹相位诊断](saved_trajectory_phase_diagnostic_20261028.md)关闭，明确原时序/共同相位误差、相位单调性及 5→6、8→9 的几何变化；离线最佳相位不进入在线控制。
3. 据此记录是否执行：路径滞后支持检验额外时间；偏离整条路径会收窄“仅时间分配足够”的假设，但不否定提前联合规划；插入段近静止则注明区分能力弱，可不执行，不暗换插入点。下一轮再实现、QA、冻结来源。

## 为何不再简单保持或复用旧 governor

旧 governor 根据几何速度和上一 admitted TCP 误差调时，没有 q/v、预测制动或裁剪条件，见 [代码](../../src/s2a/reference_governor.py:83)。其 7200 周期中 3704 暂停、4130 bridge（重叠），任务均失败，见 [历史](record_20261021.md:65)。[v2](../../src/s2a/reference_governor_v2.py:31)已修复桥接自锁/开度意图；不能重复普通暂停/桥接作为新机制。

后来的 [恢复代码](../../scripts/probe_tracking_v5_reference_recovery_native.py:157)在推进失败后将全部目标设为 cursor 8、twist 为零，没有在 hold 目标中优化下一点时间分配。[16 个真实保持周期](../../reports/reference_recovery_native_20261028.md:7)后推进仍 0；[单周期保持](reference_hold_20261028.md:5)改善 held 误差却恶化原 timed 下一点。故检验更早的联合规划，不再延长等待；不将此描述为唯一因果解释。

## 唯一成对干预

两组同 actual368、已生成 chunk46、actual367 command/rotation latch、零自身误差 profile；不用后来真实 q/v/动作。沿用状态更新 25×2ms 响应、当时固定 finger q/v、原参数/TCP 及模型局限。

| 组别 | 13 个物理端点对应的原参考相位 | 增加的 50ms 所在段 |
| --- | --- | --- |
| A：提前分配 | 0,1,2,3,4,5,5.5,6,7,8,9,10,11 | 5→6，在当前预测最紧制动项之前 |
| B：临近分配 | 0,1,2,3,4,5,6,7,8,8.5,9,10,11 | 8→9，在当前首失败目标之前 |

两组同为 650ms，原 H12 为 600ms；包含同样 12 个原节点，依序消费，后缀从 12 开始。插入点用 world 平移中点、SO(3) 最短弧中点，开度共享相位；无 hold、自由 alpha、重锚、跳点、丢后缀或 requery。原 chunk46 的 cursor 0..12 opening 均为 1.0，此段不新增开闭事件；电机状态及原命令保持语义仍登记。[chunks.json](../../outputs/tracking_preview_wam_development_v5_20261028/development/cases/task0_init0_s42_terminal058/simulation/8400_masked_position_hold/task0_init0_s42/chunks.json:599709) 重新绑定 SHA。

## 唯一 warm、约束与预算

唯一 warm 来源是原 H12 `warm[0:12,0:6]`，不用 prefix selected。A 在 u5/u6 间插入两者均值，B 在 u8/u9 间插入均值，其余逐值保留。凸性及相邻差减半保留 box/rate（包括 actual367→u0），仍需数值核对；不重新缩放、裁剪或添加第二起点。

插入会改变后续 q/v/latch，动态安全必须完整重算 H13/325 内步及九组约束。除完整 witness 的快捷返回外，warm 必须 full audit 合格且 nonlinear safety slack≥1e-8。否则本组 `warm_admission_failed`，归档原因后停止，无 restoration/第二起点；只说明初值未获准入，不是搜索失败或物理不可行。另一组可按同规则完成，但单组准入不等于配对性能胜负。

每组最多 4000 唯一完整响应/60 共享迭代、单边差分 1e-4，warm 计入预算；预检/solver 缓存复用明确登记，不隐藏重复响应，stage 不重置。总上限 8000 响应/120 迭代/2600000 内步，不能称与单次原 H12 同计算成本。初始 prefix 从 H13 实算，不强迫为 6；全部 78 命令可调整。

沿用 [H≤16 native](../../src/s2a/native_goal_trajectory.py:23) 和 [既有 prefix solver](../../src/s2a/prefix_preserving_trajectory_plan.py:42)，不新增变体。全部 13 端点 caps 为 0.01m/0.05rad；325 内步保留 0.05rad+profile，13 个 `q+0.1v` 保留 0.05rad 裕量，最终 q7dot≤0。原 box/rate、严格 pose 判定、审计容差及优化缓冲不变；success 不代替 witness。

## 两种合同必须同时记录

新候选命名为 `retimed_path`；在每个实际 50ms 端点并列计算：

- `original_timed`：对原 chunk46 cursor 0..12 依次评分，使用原 0.01m/0.05rad，13th tick 明确对原 cursor 12；不得改成对 retimed cursor 11。第 13 个原目标只作名义评价，不是两组路径规划的新增目标。
- `retimed_path`：对上表固定 13 相位的目标评分，仍使用原空间 caps；另外单列全部 12 原节点在其注册到达端点的误差、连续原节点进度、最终消费 cursor 11 和未消费后缀 12..39。13 个控制端点不计为消费 13 个原节点。

归档物理/原目标时间、共同相位、两套逐点误差/布尔值、全部裕量及裁剪。retimed 不能代替原 50ms 指标；650ms 微实验不增加全任务预算，将来闭环额外耗时须从固定预算扣除，未执行参考计未完成。

## 结果解释与下一层证据

A 通过而 B 未找到 witness，支持“需在较早受限段前分配时间”的局部线索；B 通过说明临近 8→9 的额外时间在本模型中可能已足够；两组通过只说明在本对照中增加一次 50ms 可得到新 path 合同的模型解，不能升级原 timed 资格。两组有限搜索均未找到 witness，不是动态不可行证明。若有 warm 准入失败，先按上一节单列，不能套用这些搜索成败结论。

独立 checker 逐条重构全部 H13 响应，核 native 目标、325 内步、九组约束、两套误差、阶段/预算、严格 prefix/选择，不导入生产 solver/audit 自证。新入口冻结源/输入 SHA、环境、owner/child 和实际 wait/reap；组件 QA 后方可登记执行。

完整 witness 独立通过后，另登记有界 13 周期原生物理及独立重放，核实际 cursor 9、安全、两套误差、task/grasp 和后缀，不直接派发动作。局部成功不等于任务成功；原 18 条正确参考、12 条 WAM 开发、最终 24 条新初态及末 40 点门槛保持，见 [总标准](002_wam_stability_20261021.md:10)。

截至本文编写，仅完成源码/保存报告/原 chunk 的只读审视及计划编写；没有新增响应、物理周期、世界模型调用、GPU 任务或 fresh 案例。

只读前置更新：相位诊断已actual exit0关闭（代码3e99634，[结果报告](../../reports/saved_trajectory_phase_diagnostic_20261028.md)）。prefix selected的cursor9..11均不在已登记路径的原空间caps内，不能把保存动作仅换相位视为修复。本对照据此只检验“新增一次50ms并重新联合规划动作能否改变预测轨迹”，不宣称原失败只是滞后。5→6为4.01546mm/0.0506273rad，8→9为2.95869mm/0.0592341rad，均非静止；保留原A/B插入位置，待完整prefix checker关闭后可实现并执行原有界对照。本次更新未运行新响应。
