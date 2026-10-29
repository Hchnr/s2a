# 精确TCP位姿约束候选的原生物理结果（2026-10-27）

上一回合完成旋转退化定位、精确TCP/SO(3)约束及两个保存状态探针，属于实际进展。本回合按 `plans/v0.0.1/joint_limit_guarded_physical_20261027.md`执行固定case00/01，先各1步协议核验，再各40步。新增306个20Hz控制周期全部执行，无新世界模型调用或fresh。

## 实现与实际退出

- 新 `src/s2a/joint_limit_guarded_native.py`在旧proposal之后加同状态精确TCP平移与旋转误差不高于原种子的约束；保留目标权重和原固定控制器。零旋转解直接回退原reset种子，再提交最终执行命令的限位预测，不在紧约束解上加扰动。无限位时逐位保留原命令。
- 新独立worker/owner/checker分别为 `run_joint_limit_guarded_branch.py`、`execute_joint_limit_guarded.py`、`verify_joint_limit_guarded.py`。旧候选与来源原样保留。冻结来源包括原完整运行清单、原四个新数值模块及新增两个guard模块、worker/owner/计划。
- 相关9项CPU测试通过（会话88380实际退出0，1.98秒）。当前新增文件ruff check/format通过。
- 监督会话25444实际退出0，四阶段实际wait0/reap和来源关闭均成功。case00_h40监督耗时235.34秒（含环境、88步前缀与采集），明显未证明20Hz实时计算；没有独立逐决策延迟测量。
- native独立核验会话37011退出0；逐项检查306预算、初态SHA/motor、候选共同首步、实际退出/同身份进程消失、来源SHA、命令box/rate/reset、记录中最终模型位姿约束、实际goal提交，以及从保存仿真状态重新计算TCP误差。产物 `outputs/joint_limit_guarded_verified_20261027/verified.json` SHA256 `0c64f25ee0f3ff9d42fdd65cbfe99bce0f71a7ab2ce61b293876d8336a3a5643`。

## 三方实际状态比较

| 案例/方法 | 位置RMSE(mm) | 旋转RMSE(rad) |
| --- | ---: | ---: |
| task0_init0_s42_raw / baseline | 60.243687 | 0.210714967 |
| task0_init0_s42_raw / unguarded | 18.946711 | 0.253240500 |
| task0_init0_s42_raw / guarded | 48.021567 | 0.202159652 |
| task0_init0_s42_terminal058 / baseline | 0.784190 | 0.002342369 |
| task0_init0_s42_terminal058 / unguarded | 0.784190 | 0.002342369 |
| task0_init0_s42_terminal058 / guarded | 0.784190 | 0.002342369 |

case00相对原基线的平移误差约降低20.3%，旋转约降低4.1%；新约束避免了上一候选明显的旋转交换，但未保留其全部平移收益。40周期中35周期选择约束修正，5周期回退同状态原种子。原proposal优化响应评估总数4375，guard优化额外4717；该统计不包含额外名义预测校验，不应误称全部模型函数调用次数。

case00最终开度0.04424529，目标bowl抓取接触步数0，任务成功步数0。无限位case01全部40步原记录字段与旧基线完全一致；独立三方比较及该逐字段检查保存在 `three_way_comparison.json`。

## 决策与未完成项

局部实际平移和旋转都改善，但绝对误差仍大且无任务成功。该世界模型参考局部分支不是原18条正确参考资格或完整600步政策，因此不将其写成18/18通过或失败；两项完整验证均尚未执行。当前不进入fresh，仍未达到总目标。

下一步应保留原四周期预览范围，在关节状态空间显式展开多周期实际goal重置与单边限位，避免依赖单周期局部交换；先验证展开、梯度和未来参考因果性，再做同一固定物理比较。不能仅调低旋转权重或放宽最终门槛。原18/18正确参考、完整WAM开发及最终24条留出新初态要求保持。

全仓静态检查另有两项已有失败：`validate_temporal_response.py`导入顺序、`audit_rotation_goal_precision.py`格式；git状态显示它们没有本次工作区改动。本次没有自动改动历史脚本，不能声称全仓ruff通过。当前新增文件检查通过；未运行全套pytest，只执行上述相关9项测试。
