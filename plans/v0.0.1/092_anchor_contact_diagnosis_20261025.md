# 092 六anchor接触与命令力矩灵敏度检查

091冻结M/J诊断22/24项改善，但group4旋转/角速度失败。使用086已有baseline XML和anchor状态，mj_forward重建接触几何，不运行mj_step。不将默认零ctrl下的重算约束力当作历史真实接触力。记录全部接触geom/body、距离、约束维数及是否连到robot/gripper；六anchor同口径。计算J^T blockdiag(Lambda_pos,Lambda_ori) Kp对normalized translation命令的关节力矩导数，与实际XML actuator限幅比较，明确导数不能证明实际饱和，历史controller缓存和空域初态尚未复原。

核验实际SITE pose/twist、M正定、来源SHA和原采集生命周期证据。诊断只用历史完整状态，属于离线分析，object/contact真值不得进入在线S2A。无拟合/物理积分/世界模型/fresh。结果不授予控制资格。实际uncoupling字段仍未直接测量；后续在合格独立实例中读取，不修改共享环境。
