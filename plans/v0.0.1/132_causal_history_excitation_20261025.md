# 132 训练anchor之前的因果命令激励

只使用116全部66训练anchor的原生前缀，固定最近最多20已完成周期；不使用anchor本周期或probe未来。计算normalized translation command中心化矩阵及增量矩阵奇异值，记录数值秩与最小/最大奇异值比例。数值秩用max(shape)*eps*smax，不以人为阈值选择控制器。周期不足显式记录，不补未来。此为局部在线辨识必要条件诊断，不证明足够/因果传递稳定。无新fit/物理/模型/fresh，原门槛不变。
