# 207 直接arm逆模型的因果输入合同

输入是当前实测SITE/关节/手指/电机/previous goal/实际过去命令/已完成创新/wrench，以及显式已生成的下一期望SITE pose和已决定的当前gripper command。清除目标arm action及其派生action_goal_delta；从原89条件删去当前rotation command goal和spring六维，保留83可用条件，再加desired-current世界平移3和Log(Rdesired*Rcurrent^T)3，组成新语义raw89。不能复用旧146统计或系数。

训练时真实下一状态可作为显式期望参考，不是当前实测反馈；训练target arm action永不进入特征。direct_arm_inverse_inputs.py与3项需求测试已实现，覆盖teacher action/派生goal/未来label无影响、单个desired参考只影响对应行、SO3/世界坐标和显式gripper输入合同。此仅输入与泄漏检查，无响应拟合或控制资格。
