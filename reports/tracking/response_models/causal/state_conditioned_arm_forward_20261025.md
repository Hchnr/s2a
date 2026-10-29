# 088 条件化arm前向模型：核心实现，尚未拟合

[088登记](../plans/v0.0.1/088_state_conditioned_arm_forward_20261025.md)固定只用旧24fit/4865行、每轨迹等权、FP64 ridge0.001及71条件/288设计/12输出，无参数扫描。6holdout和086探针不进fit或尺度统计；全部预测和局部响应门槛通过后也仅为离线forward资格，不是控制成功。

核心实现src/s2a/state_conditioned_arm_forward.py，4项数学/因果检查通过，见tests/test_state_conditioned_arm_forward.py；条件仅含当前状态、因果历史及已知当前action目标，平移action保持仿射。motor未来目标由当前cache和计划夹爪命令递推，不是未来实测值。

尚未执行fit、独立数值复核、holdout预测或086局部响应验证；0新物理/模型/fresh，无新控制收益。总目标与原18/18、WAM开发及最终24条验收不变。

## 088唯一拟合及完整关闭结果

唯一4865行/24fit参数拟合完成；4项CPU检查、真实首164行独立特征对照及导入准入通过。独立全特征/尺度/标签/权重和增广SVD复算通过，参数最大差2.397e-15、正规方程梯度最大4.366e-18。生产拟合1次、独立数值重算1次，未自动重fit。

6holdout共1315行的position/SO3/线速度/角速度全部24项不劣名义PD；但086局部响应仅17/24项通过，失败group0和2的位置/线速度、group4的SO3/角速度、group5的位置。离线forward资格False，不部署逆控制、不进入WAM/fresh。三阶段及owner实际wait0/reap、无信号与来源不变均核验，closure_verified.json及validation_result.json保存原输出目录。

这表明仅使用原生轨迹的预测资格不足以授予正确局部命令响应。下一模型若引入成对响应，必须预先固定训练/验证anchor划分及权重，不把所有已见探针混入fit后再声称独立验证。0新增物理/世界模型/fresh，总目标未完成。
