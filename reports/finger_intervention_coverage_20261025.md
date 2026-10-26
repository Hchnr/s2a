# 187 手指训练干预覆盖审查

`outputs/finger_intervention_coverage_20261025/verified.json`完成69唯一训练anchor与3未训练anchor当前状态/几何核验，保存原始来源。每点只恢复原raw state并mj_forward，实测手指q与原数组1e-12一致；无积分、无历史力重建。69训练点中40有手指几何接触，15点motor目标gap<=5mm而实际gap>=5mm；这些阈值只作描述，不当作接触/受载真值。

训练手指涉及body计数（可重叠）：bowl9、wine_bottle9、cream_cheese10、table11、wine_rack6。bowl接触9点中含motor目标0且实测gap约6.85–15.11mm的多个点。因此不能声称训练中没有物体受载或完全缺少该gap范围。

未训练组1实测gap8.502mm、motor目标0，左右手指均与bowl接触；组2gap79.873mm、目标80mm且无手指接触；组4gap80.371mm、目标80mm且左指触桌。三个局部失败组跨受载、空载和桌面接触，不能只增加一个“loaded”标签就宣称可解决。后续应比较状态距离/接触模式下真正的干预响应，并保留全部独立划分，暂不盲目采集与现有数据重复的受载样本。

176因果半权重回归已完整闭合：18条4907周期、36stage与owner正常wait0/reap及来源核验，稳定17/18，但误差与稳定综合资格15/18；case03任务失败、case06位置误差失败、case14位置/旋转失败。两种计数分开，不能把17条稳定说成17条控制资格。178仍在运行。没有WAM/fresh资格，总目标未完成。
