# 072 当前电机状态条件逆模型：单次拟合完成，离线资格失败

按固定30训练轨迹24fit/6辨识留出、FP64 ridge0.001及每轨迹等权完成唯一一次拟合，共6180行。输入为当前实测开度、电机目标、过去开度差与已生成未来1/2/4/8开度参考，明确availability mask；7连续量标准化及二阶展开后40维。目标为本周期实际motor目标，不将实测opening直接当motor目标。训练future只作desired reference，在线API无未来观测/对象/成功输入。

特征/窗口/限幅/可达原生命令并列规则6项实际测试、Ruff通过。所有30输入及原roster SHA读取前后复核。产物`outputs/state_conditioned_gripper_inverse_20261025`保存参数、完整fit_cache、行组/权重/来源及逐轨迹误差，0模型/物理/fresh。

五条留出motor预测与可达量化误差优于保持当前motor基线；wine/demo_39退化：预测RMSE0.10232、量化可达RMSE0.08002，保持基线0.07157。原要求每条不劣未满足，offline_performance_eligible=False、physical_qualification=False。不接入物理或WAM，不扫描lambda/特征追赶同一留出案例。其他五条量化误差约0.01705–0.02621，保持基线0.06118–0.10230。

这些是motor目标误差，不是前向state或任务成功。尚需独立重建特征/目标、QR与逐轨迹核验，排除拟合/索引实现问题后再决定下一假设，原控制资格与最终任务门槛不变。

## 完整独立数值核验已通过，性能失败仍成立

`verify_state_conditioned_gripper_inverse.py`不导入producer特征或动作函数，逐scalar重建全部6180 raw特征/availability/目标/组号/权重，与fit_cache逐值完全相同；独立重建标准化/40维展开，以增广最小二乘而非原正规方程求解。系数最大差1.02e-12，增广梯度最大3.38e-15，所有逐轨迹预测/量化/保持基线误差复算一致，来源和fit SHA前后不变。Ruff及实际执行通过，independent_verified.json numerically_verified=True、offline_performance_eligible=False。

因此当前留出失败没有被索引或求解错误解释，不将数值通过当作模型性能资格。后续可有限比较直接目标与相对当前motor的残差目标：二者有不同正则化先验，保持当前目标是已知强基线；若另试，须先登记单一残差对照，不扫描同一留出直到通过，并保留已被用于诊断这一事实。
