# 208 待执行：直接平移逆的独立辨识对照

保持原24fit/6holdout及固定控制接口。使用原4865自然周期及已资格72训练anchor的7支绝对分支，共5369训练行；原086未训练1/2/4不进入训练。每行当前条件来自prefix端点，显式desired SITE为该训练分支实际下一pose，target为实际sent normalized translation3；这是期望参考监督，不是未来观测反馈。当前gripper command显式提供，teacher arm command和派生goal按207清除。

解析位置PD逆作为独立nominal，当前twist与固定kp/kd/dt及实际affine接口求translation，保留原边界。新raw89统计仅从24fit4865自然行得到，固定seed42的64组sin/cos与线性项组成218维残差基，ridge=.001、仅常数不罚；拟合target minus nominal。自然总权重.5、504分支总权重.5，唯一固定拟合，不挑多组超参。独立标量特征、解析nominal及增广SVD复核。

原6holdout动作RMSE仅诊断，绝不等同状态控制成功。仍须原固定18条经过实际控制器的完整状态误差/稳定任务及生命周期审查，之后WAMdev和最终24新初态。原旋转/夹爪策略若保留须显式记录，不宣称学习全action或已解决接触。当前只有输入合同，未构建数据或拟合。
