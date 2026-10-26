# 089 固定差分监督：完整关闭，未训练响应资格失败

使用原24fit/4865行及固定groups[0,3,5]的9条中心差分方程，原生/差分总权重各0.5，ridge0.001、71/288/12设计不变；统计沿用原24fit，groups[1,2,4]未参与拟合。4项核心检查、全部实际9方程独立组装对照及导入准入通过。

唯一新参数拟合1次，独立增广SVD复算1次，系数最大差4.36546e-15、正规梯度最大1.09344e-17，数值通过。6holdout四类预测24/24通过；训练anchor响应12/12，但未训练anchor9/12（group2位置，group4 SO3/角速度失败），全部响应21/24。offline_forward_eligible=False，不部署控制。

三阶段及owner实际wait0/reap、无信号和来源不变审查通过，outputs/contrast_conditioned_arm_forward_20261025/closure_verified.json和validation_result.json保存依据。没有把训练anchor计作独立验证；未扫描参数或自动重fit。0新物理/世界模型/fresh，总目标未完成。
