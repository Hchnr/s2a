# 183 原六anchor手指局部arm干预斜率

冻结181唯一系数，在原086六anchor各7支实际已审查干预上独立重建current条件、实际command/goal与delta间距SI米，计算三轴central per-normalized arm command导数。组0/3/5用于contrast fit，1/2/4未训练。生产1128设计与独立设计预测1e-10核对；保持当前/历史/当前wrench原因果协议。

每组模型三轴导数误差范数须不劣于零arm耦合基线（固定相同gripper command、当前速度和初态，baseline对arm导数0），不添加容差放宽、不因结果改正则/数据。分别记录训练/未训练。自然holdout过关不足以替代此门槛，局部失败不接入控制。无新物理、拟合、WAM或fresh，原总目标与门槛保持。
