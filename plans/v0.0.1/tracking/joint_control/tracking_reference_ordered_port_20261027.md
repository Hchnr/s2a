# 恢复原参考坐标变换运算顺序：case06基线复现

当前候选在原case06（wine/demo46）任务稳定门槛失败，不能进入WAM/fresh。此前首条入口复现不能保证全部案例保持原数值轨迹。

从原代码核实原变换为 `targets @ inverse(source_from_tcp) @ source_from_site`，新入口合并为 `targets @ inverse(inverse(source_from_site) @ source_from_tcp)`。两者几何等价，但浮点运算不同。在原生Python/NumPy环境与8线程下，新入口首命令原差4.440892098500626e-15，恢复原顺序后差0。诊断原产物 `outputs/tracking_reference_roundoff_native_20261027/diagnostic.json`。早先使用.venv的两次探针不能作为原环境逐值复现证据，其产物保留。

独立版本 `run_tracking_correct_reference_ordered.py` 仅恢复两处生成SITE/anchor的矩阵乘法顺序，其余控制算法、阈值、观测、夹爪和参考调度保持；旧来源不修改。通过 `execute_tracking_ordered_port06.py` 在CPU/OSMesa执行原case06 baseline完整334步，之后 `check_tracking_reference_ordered_parity.py` 与旧合格基线逐端点/动作/目标/任务谓词核对，仍使用绝对1e-8容差、相对0。无WAM/fresh，无自动重试，预算334物理周期，独立比较不增加物理步。

注册：`outputs/tracking_correct_reference_ordered_port06_20261027/registration.json`，结果写独立目录 `outputs/tracking_correct_reference_ordered_20261027/case06_baseline/`。本次仅验证入口修正能否恢复该基线，不是候选18条资格，更不是限位优化收益或接触鲁棒性证明。
