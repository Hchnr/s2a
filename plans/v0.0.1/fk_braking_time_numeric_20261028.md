# 实际 cursor9 停止点的投影制动时间敏感性（CPU，2026-10-28）

固定默认 H8 分支的真实 endpoint377。原 [停止约束归因诊断](fk_stop_constraint_audit_20261028.md) 保存候选的首周期 q7 上界制动约束处于活动状态；H1 去除未来周期和 terminal 方向后，仍未找到满足原 pose 门槛的候选，最好已保存线性可行候选误差约 .00998997m/.07307213rad。这支持检查额外投影条件 `q + tau * qvel` 对有限搜索结果的影响，尚不能证明它是唯一原因，也不能证明物理不可行。

本次只实现两个新的数值搜索，本阶段不派发动作。执行前必须读取 `outputs/tracking_v5_fk_stop_constraint_audit_v2_check_execution_20261028/result.json`，确认独立 v2 审计 owner/result 全部 complete/passed，以及实际 wait0/reap0。经其注册引用固定原 audit producer、原 `no_terminal` / tau=.1s 基线与默认 H8 native 分支；旧失败及其检查记录保留，不重跑或覆盖。

在线条件仅来自该分支真实 endpoint377 的 q/qvel、当前 M/J、固定控制器参数与 XML 几何、此前九个已完成周期的自身误差 profile、实际上一命令，以及 step368 已生成的 chunk46。目标固定 cursor9；twist 仍从完整已生成参考使用邻点8/10计算。不得混入 H12/H16、其他旧分支或未来实测状态。

| 模式 | H | 投影制动 tau(s) | terminal normals |
| --- | ---: | ---: | --- |
| 原归档 `no_terminal` 基线，仅引用 | 1 | .1 | 空 |
| `braking_050`，唯一新搜索 | 1 | .05 | 空 |
| `braking_025`，唯一新搜索 | 1 | .025 | 空 |

两个新搜索使用同一当前状态、profile、实际上一命令作为 seed，以及原 H8 最近线性可行候选的首命令作为 supplied warm。保留求解器实际 LP 替换与 start policy，不能把相同 supplied warm 描述为最终优化起点必然相同。每组160响应/60迭代，共320响应，seed42；不追加参数、不重试、不依据结果调节预算。

除明确改变投影制动时间外，原首周期 pose .01m/.05rad、动作 box、每周期 .25 rate、全部25个2ms内环的 .05rad 加自身历史 profile 裕量、固定50ms物理时间与原 pose/twist 成本保持。投影制动约束自身的关节边界仍留 .05rad。两个新搜索都没有未来周期与 terminal 方向，比较基线也是相同 H1/no-terminal 政策；不得把它们称为完整 H8 安全计划。

搜索前先按 tau=.1s 重建原 H8 保存候选的完整冻结 joint rollout、FK/TCP误差及全部原约束余量，核对其已保存误差、profile裕量及当前状态/动力学。原 H8 使用原 terminal normals，不改变其历史合同。每个新返回候选再用与该次搜索一致的 tau 重建 rollout、误差与分组余量；不能用 .1s 审计 .025s 候选，也不能把较短 tau 的通过标签套到原合同。

产物登记独立 v2 owner/result、原 audit producer、原 no-terminal 基线完整 entry、控制/XML/参考与代码指纹。每个新 entry 保存 mode、braking_time_s、实际求解耗时、answer、完整 candidate/rollout/errors/约束审计，未找到 witness 的结果与最好已检查尝试同样保留。0物理步、0世界模型调用、0fresh，GPU继续禁用。

入口：`execute_tracking_v5_fk_braking_time_numeric.py --launch`；新 producer/owner/supervisor 分别写 `outputs/tracking_v5_fk_braking_time_numeric_20261028`、`outputs/tracking_v5_fk_braking_time_numeric_execution_20261028`、`outputs/tracking_v5_fk_braking_time_numeric_supervisor_20261028`。独立检查器另行实现，须复算当前状态与动力学、FK误差及各候选自身 tau 下的全部约束；结果尚未执行或核验时不计为新增证据。

即使存在新 H1 模型 witness，也只说明额外投影制动时间的数值敏感性。任何后续实际 CPU feedback 都需要新的明确预登记与完整未来安全验证，并继续保持原实际 .05rad 关节裕量、真实 pose 门槛及停止规则；本计划不构成实际控制安全证据。原18条资格、完整12条WAM开发、final24新初态与最终稳定任务门槛保持，任务目标尚未达成。

两次新数值搜索及独立检查已actual wait0/reap0完整关闭：.05/.025s旋转误差分别.0698699/.0663121rad，均无首点witness；166响应，物理/模型/fresh0。独立十个构型和全部各自tau约束通过，未取得实际执行或任务资格。[完整结果](../../reports/fk_braking_time_numeric_20261028.md)。
