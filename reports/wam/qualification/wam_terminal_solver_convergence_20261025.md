# 058固定末节点目标：数值资格通过，性能待验

原057生产与独立审查失败保留；058是另行登记的固定CPU求解实验，未改变训练数据、目标、λ、ε或资格阈值。计划见 `plans/v0.0.1/058_terminal_solver_convergence_20261025.md`。累计生产fit尝试2次，成功1次；不是原057自动续跑。

058仍从052头开始，在原300 D2末节点/12000行上求解，同一052固定朝向、feature center/RMS、相同H8/H16/H40权重。位置CG351次、info0；夹爪固定100轮IRLS全部子问题CG info0，末轮134次CG，相对子问题残差9.4057e-10。唯一调整为原计划外另登记的100轮预算，不使用057失败参数热启动。

实际producer PID3034412/birth580351982，工具session97664最终exit0，/proc同PID已不存在。生产 `fit_result.json` passed=True。独立脚本完整逐轨迹重算12000行位置梯度及原非平滑L1可行对偶界，工具session82934实际exit0，`fit_verified.json` passed=True：

- 位置独立相对梯度残差7.587437717092629e-10，小于原1e-7。
- 夹爪原始-对偶gap1.6456282574545122e-6，小于原2e-4。
- rotation六列、center、scale、ridge及schema与052逐值保留；生产和审查消费输入前后SHA均不变。

新参数：`outputs/terminal_solver_convergence_20261025/parameters.json`，SHA256 `30210a2bda69962bbd24f8e86849f58c4516d2209bfb3d8a5903fc45ae543b4f`。生产及独立审查都只认证离线条件凸目标；`performance_qualified=False`。

`src/s2a/terminal_readout_policy.py`实现统一分派：实际scheduler索引0–33选052，34选新头；拒绝错误索引、rotation或feature变更。Mixin从原capture bridge实际record读取step_index，调用原父实现并在异常时恢复readout，不用单独计数器。三项CPU测试覆盖35节点分派、rotation篡改拒绝及原桥异常恢复；Ruff通过。该组件尚未接入实际GPU sampler，不能视为完整推理认证。

下一步接入真实sampler、固定参数和正常退出资格，核验末节点策略/精度/历史同一性，再另登记原同步state validation；通过原state门槛才进入开发。原24条fresh稳定≥20/24、各任务≥4/8及实时/真机另验保持不变。当前新模型调用、物理、fresh均为0，总目标未完成。
