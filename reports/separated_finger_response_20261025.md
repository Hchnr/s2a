# 185 手指斜率/偏置两阶段结果

按185固定设计，以207真实干预拟合846维arm斜率，再固定斜率用4865自然样本拟合282维偏置，ridge=.001和原权重/统计/频率不变。生产两阶段与独立增广SVD一致，各阶段秩完整，fit/verify owned stage实际wait0/reap、根run工具正常exit0。数值凭据见`outputs/arm_finger_separated_fit_20261025/independent_verified.json`。

原六自然holdout均优于间距保持、五条优于当前速度基线；但未训练局部组仍为0/3通过，训练组5和未训练1/2/4失败，总计2/6。未训练组1模型误差.00384189m/normalized input对零基线.00248347，组2模型.00171974对.00000112，组4模型.00136927对.00081463。结果在`outputs/separated_finger_holdout_20261025`和`outputs/separated_finger_local_response_20261025`。候选资格失败，不进入控制，不调低门槛。

首次入口在source roster创建前因迁移脚本错误引用不存在的计划路径退出，未创建fit输出或执行任何拟合；已修正为实际185计划后才执行上述唯一两阶段辨识。旧181结果/来源未改。下一步先分析真实plus/minus/baseline曲率和响应模式，不能仅凭两阶段数值过关声称斜率可信。新增物理/WAM/fresh均0，原总目标与验收保持。
