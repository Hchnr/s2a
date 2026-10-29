# 057末节点候选准备

固定方案见 `plans/v0.0.1/057_terminal_trajectory_objective_20261025.md`。仅用D2的300请求末节点、150训练锚点/12000行；基线为052，不混合不同策略历史。部署拟定节点0–33和所有rotation输出保持052，节点34更新xyz/opening四列。尚未实现或认证实际sampler时序。

`prepare_wam_terminal_trajectory.py`已实际执行。先核验原聚合准入（包含完整D2数值与生命周期），然后逐请求重建raw C0并与保存数组精确一致，检查052预测残差与已部署保存残差在原rtol1e-9/atol1e-11内一致。只读取原150训练窗口，朝向由当前实测态及052预测增量积分，未来实测态只用作标签。输入前后SHA不变。

输出：`outputs/terminal_trajectory_objective_20261025/prepared.json` 和 `training_cache.npz`。缓存126257560字节，SHA256为443f975f1e95ea5c1d264feb2f36e6587a0ab530cb57f3ac3b31e161cdf9a687。另行重读检查全部数组有限、phi为300×40×1291且bias恒1、端点权重和为1、真实开度范围合法。未写新参数，producer_fits=0。

`src/s2a/terminal_objective_solver.py`实现矩阵自由ridge二次CG与固定ε平滑LAD的IRLS；最终证书针对原非平滑L1目标。对偶可行条件|s|≤w，对偶值sᵀb−||Aᵀs||²/(4λ)，保留每次CG实际迭代数及normal residual、IRLS每次原始/对偶目标及gap。3项CPU测试通过：CG与独立dense线性解一致；LAD已知中位数最优解及对偶夹逼；非最优输入对偶界检查。相关Ruff/格式通过。

尚缺完整缓存到矩阵算子的生产接入、独立完整数值审查及持久化执行入口，未进行生产fit。计划固定λ/ε/预算与容差，不按validation调整。新模型、物理、fresh均为0；原056性能失败、最终24条门槛及实时/真机另验保持不变。

## 完整算子与实际单次fit结果

后续已接通 `src/s2a/terminal_trajectory_operators.py`，并执行完整300请求/12000行核对：与逐轨迹独立实现相比，位置/开度损失绝对差4.44e-16/5.55e-17，梯度最大差5.55e-16/6.63e-15，两个伴随内积相对差均小于1.4e-15。产物为该namespace下 `operator_qualification.json`。相关算子、求解器及链式梯度共9项CPU测试和Ruff/格式通过。

本次唯一生产fit已实际执行，CPU空CUDA/LD、8线程，PID3031555/birth580339167，工具session64335最终返回exit1；不是观察超时。位置CG351次、info0、相对normal residual7.5874e-10。开度完成固定30次IRLS且全部CG info0；最终原始目标0.07321725828915376、可行对偶界0.07046119541731943、gap0.0027560628718343305，超过预定2e-4上限。因此 `fit_result.json` passed=False，`fit_failure.json`保留，未追加迭代或修改容差。

独立审查脚本随后实际完整重算逐轨迹目标、梯度及L1对偶证书，`fit_verified.json` passed=False且实际exit1；失败得到独立确认。新参数虽保存于 `parameters.json`，不作为合格候选发布，不进入模型或validation。原来源、预算、失败和参数均不得覆盖。此结果说明固定优化预算未获得要求的数值证书，不能据此断言新目标的任务效果好坏。

下一步应依据实际迭代轨迹明确新的数值收敛方案及独立预算，再另行登记；不得静默续跑本次失败或自动增加迭代。新生产fit累计1次、成功0次，新模型/物理/fresh均为0。
