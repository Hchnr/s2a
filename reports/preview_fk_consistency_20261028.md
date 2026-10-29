# 已接收预览解的内部运动学不一致（2026-10-28，CPU）

[登记检查](../plans/v0.0.1/preview_fk_consistency_20261028.md)固定同一实际step368及已保存的完整8点模型witness，不重新优化或执行动作。将每个预测joint_position代入独立compiled XML标量关节FK，比较其TCP与同一模型保存的Cartesian pose和原目标。

**原模型端点精度标签8/8合格，但同一预测关节构型的独立FK只有5/8满足原.01m/.05rad精度。** 第6/7/8端点FK对原目标的位置误差约.0118375/.0142268/.0182082m；最后旋转误差.0503260rad也超过.05rad。模型Cartesian pose与同q独立FK最大内部差异.01701236716m/.04484467660rad，不能将原8/8模型标签当成实际机器人末端跟踪保证。

这不是实测未来轨迹误差：FK只检查模型自己预测的q是否与它自己的末端pose一致，不包含控制器执行、接触或未来实测state。检查没有证明此缺陷是所有任务失败的唯一根因，也没有新增任务成功。

初始实际q的XML SITE pose与原当前SITE一致；独立XML中心差分对当时实测world SITE Jacobian的最大差1.50256529e-10。对应模型源码`src/s2a/interior_joint_preview.py:104`用冻结当前J乘joint增量后累计Cartesian平移/SO(3)旋转，同时另存累计joint_position。当前点正确的Jacobian不代表较远预测构型也可用冻结Jacobian递推精确FK。这解释了为何必须核验整段预测的joint/TCP一致性，而不能仅靠优化预算或窗口长度。

owner actual wait0/reap0，`outputs/tracking_v5_preview_fk_consistency_20261028/verified.json` SHA `1d06af338652019c1c8d1624e5ee6dee9b5c5af0d557d0a23e9b98d545d00b03`；八点完整比较另存rows.json，source/input指纹可追溯。新增物理、模型调用、fresh及优化搜索均0，Ruff/format通过。这里complete/passed仅表示审查协议正常结束，结果是在模型中发现不一致，不是模型控制资格通过。

下一候选应从预测q的实际FK重建每个端点pose和world twist，连同正确Jacobian/Hessian链式导数计算同一pose/twist目标；继续保留原关节/制动/terminal、参考时序、因果历史及有限预算。仍冻结free M/J时需明确剩余动力学近似，不能因修正几何就宣称接触、执行或任务稳定性。先验证非交换旋转、TCP偏置及运动学/目标梯度，再做同一实际状态数值比较，有完整合规模型解才进入固定接口原生反馈验证。

GPU继续禁用，原18条4907步/完整12WAM开发/final24新初态门槛不变，尚无总目标完成证据。
