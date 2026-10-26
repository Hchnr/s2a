# 183–184 手指局部动作响应资格失败

`outputs/arm_finger_local_response_20261025/result.json`完成原086六anchor×三轴central arm干预导数评价，生产与独立1128设计预测1e-10一致。模型仅2/6组不劣于零arm耦合基线：训练组0/3通过，训练组5失败；未训练组1/2/4全部失败（0/3）。组1模型误差.00356035m/normalized command对零基线.00248347；组2模型.00149979对.00000112；组4模型.00171485对.00081463。

`outputs/arm_finger_local_independent_20261025/verified.json`又从各支raw sim_states/XML独立复算实际导数，逐轴1e-12一致，重算同样2/6。尤其组2真实arm耦合近零而模型预测明显响应，不能把182自然holdout6/6通过当作可用动作斜率。181候选离线局部资格失败，不接入控制，不放宽基线或调整验证标签。

下一方向是区分由arm干预辨识的斜率与自然运动/夹爪动作相关的偏置，保持原数据划分并在新namespace显式对照；尚未拟合下一模型，也没有新控制/WAM/fresh收益。原176/178回归继续，独立checker PID163444/case15与PID163610/case10已实查start_ticks一致存活。原18/18与最终新初态门槛保持，总目标未完成。
