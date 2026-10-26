# 109 联合非线性当前反馈条件对照

候选保留107完整特征，追加固定32对当前89维状态/wrench联合cos/sin特征，条件基282、设计1128列。原5000方程、24fit统计、数据划分、两类各.5权重及ridge=.001保持。唯一候选、无扫描，不使用开发探针标签训练。

旧模型等价嵌入、action仿射、当前反馈敏感、未来数据隔离以及独立标量特征验证2项测试通过。全部4865自然行与135干预方程的生产/独立一致性已实际检查通过，新增19份相关文件静态/格式检查通过。完整持久监督入口准入已启动；尚无参数数值资格或性能结果，不声明控制收益。

计划：plans/v0.0.1/109_joint_wrench_condition_forward_20261025.md。执行：scripts/run_joint_wrench_condition_forward.py。产物：outputs/joint_wrench_condition_forward_20261025。原107资格失败及18/18、WAM开发/最终24新初态门槛保持。没有新增物理、世界模型或fresh测试，总目标未完成。

## 完整结果：资格失败

5000方程唯一fit、独立增广SVD、原holdout与局部响应验证已执行。holdout23/24、局部23/24，offline_forward_eligible=false，不进入控制。系数最大独立差8.398148712909781e-15，最大正则梯度1.5176413062174937e-16。

相比107，group4角速度响应通过（矩阵误差.21075930164045367，nominal .269341306907388）；位置与线速度仍通过。group4旋转增量仍失败：.00870075468893238 > nominal .008201913715536734。holdout28线速度RMSE .033048603436654694 > nominal .031242752751587693。局部改善不能替代逐通道完整门槛，训练表征改进未证明闭环收益。

权威产物位于outputs/joint_wrench_condition_forward_20261025：fit_result.json、independent_fit_verified.json、validation_result.json。全局fit1，新增物理/世界模型/fresh0，原总目标未完成。
