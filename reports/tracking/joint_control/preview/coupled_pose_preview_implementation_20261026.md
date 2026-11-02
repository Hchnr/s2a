# 四步预览候选的独立数值资格

247候选已完成基础实现及独立检查器，尚未接入真实仿真。原总目标、18/18正确参考门槛、世界模型开发和fresh24门槛不变。

## 本轮完成

- `scripts/coupled_pose_preview_independent.py`使用独立旋转弹簧差分/solve_ivp建立响应矩阵，再逐步传播各规划命令、位姿切向状态、速度和已完成innovation；不调用生产规划器构造凝聚预测。
- 两个固定有限差分步长复核目标梯度，以NNLS独立重建非负约束乘子，检查原幅值/变化上限、驻点残差<2e-6和互补残差<1e-8。有限差分探针不做命令裁剪，避免约束边界上的错误梯度。
- `tests/test_coupled_pose_preview_independent.py`补充真实case04/71的1–4步时域与innovation核验，拒绝首命令、预测、目标值、历史与超限序列损坏。与原组件测试合计15项通过（2.96秒）；相关五文件Ruff及格式检查通过。
- `scripts/audit_coupled_preview_extraction.py`对原三初态H1/H4、共1200周期逐周期重算抽取后的候选，首命令最大差8.8599e-11，小于预先固定1e-8复现门槛。来源SHA复核通过，输出`outputs/coupled_pose_preview_extraction_20261026/verified.json`。

## 证据范围

抽取复核使用既有局部模型状态序列，证明可复用实现与此前诊断的计算行为一致，不是额外物理闭环成功。独立ODE检查仍针对冻结M/J、identity-goal弹簧的切向模型；KKT不证明目标相关SO(3)非线性优化或接触环境最优。

下一项需要从当前查询的生成chunk独立解码参考和twist，覆盖动态参考、尾段缩短时域、当前实测M/J及已完成历史，禁止读取未来实测状态来充当在线参考；随后验证最终实际命令的原生解码与SOURCE历史提交。此轮未完成全18记录准入、候选物理回归或新WAM/fresh。

原SQP回归owner898994/start_ticks594206716与closure899090/start_ticks594207420继续同身份存活，已见前6条独立证明；完整18条仍未关闭。原CPU全套测试979621/start_ticks594533595仍执行，本报告15项通过不能替代全套结论。
