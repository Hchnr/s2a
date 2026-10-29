# 112 因果wrench变化量对照

保留109完整条件基，追加六维当前减上一周期world-at-control-SITE wrench变化量，以既有fit当前wrench尺度归一化。条件基288、设计1152列，action仿射。差分不是固定空间点的纯接触力导数，两个测量各自在当时SITE点表达。

完整轨迹先计算差分后截取anchor，首周期零；截断且缺前缀拒绝准备，缺显式变化量拒绝预测。生产端与独立端分别构造差分及特征。3项必要合同测试通过，包含轨迹重置、非零历史独立一致性、旧模型嵌入、action仿射及未来隔离。全部4865自然行和135干预方程独立一致性检查通过。

原5000方程、划分、固定统计、Fourier矩阵、权重和ridge保持，唯一候选。正式准入已启动；尚无参数数值资格和性能结果。计划plans/v0.0.1/112_causal_wrench_change_forward_20261025.md，执行scripts/run_causal_wrench_change_forward.py，产物outputs/causal_wrench_change_forward_20261025。没有新增物理、世界模型或fresh；原控制/WAM/最终验收门槛保持，总目标未完成。

## 完整结果：资格失败，不采用

唯一5000方程fit、独立增广SVD及原验证完成。holdout23/24、局部22/24，offline_forward_eligible=false。原holdout28线速度通过，但holdout19线速度退化；group4旋转增量和角速度仍失败。相比109局部23/24，新增变化量出现退化，不能选择性报告修复28而忽略19和group4。未进入控制/WAM/fresh。

权威产物outputs/causal_wrench_change_forward_20261025/fit_result.json、independent_fit_verified.json、validation_result.json。新全局fit1，新增物理/世界模型/fresh0。原全部验收门槛与总目标保持，未完成。
