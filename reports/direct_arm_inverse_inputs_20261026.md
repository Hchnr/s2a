# 207 直接逆输入合同实现

src/s2a/direct_arm_inverse_inputs.py显式区分当前实测状态、已生成desired SITE参考及当前可用gripper candidate。原teacher arm action和controller goal被覆盖清除，当前rotation goal/spring六维不作为输入；新83当前/历史条件加6维期望state差共89维，统计语义与146不同。

3项测试通过：任意篡改teacher action/派生goal与未来label不影响特征；原输入不被修改；显式desired pose的世界平移/SO3编码正确且不影响其他行；非法gripper形状拒绝。没有使用动作RMSE证明控制，没有新模型或物理收益。208只预登记数据/拟合与原实际验收要求，当前未执行。
