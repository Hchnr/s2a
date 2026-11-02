# 213 直接逆真实控制接入与独立审查

前提为209输入/标签独立proof、210唯一拟合/SVD和212原30条6180记录状态运行命令proof及其全部SHA保持。所有新文件/产物另开namespace，不修改已绑定的旧执行源码。不凭六条动作RMSE调整特征、ridge、权重或限速。

将211平移预测替换205执行器中的前向逆，保持原18profiles/4907控制周期、相同参考时钟、初态、controller、083/159旋转、079夹爪、seed42。初始实际previous command为零；当前夹爪候选先产生，作为逆输入的已知命令。当前输入只读真实sim状态，desired SOURCE转换为SITE后提供完整姿态。直接逆只作用平移，固定5/s变化率，禁止叠加支持混合、旧前向逆或预测5mm限速。

最终命令写入actual action及仿射goal_delta后，173 SOURCE residual必须commit实际执行命令；150 SITE响应创新只在env.step完成后更新。用于150的固定物理nominal保留原146物理参数；禁止用直接逆输出充当前向预测。日志分别记录direct nominal/residual/prediction/final rate clamp，以及物理nominal；不将这些伪装为旧前向逆KKT解或已验证一步状态预测。

独立checker由直接预测等式替换旧KKT审查：独立当前状态绑定、标量输入/解析逆/特征/系数乘积、实际previous history、裁剪和goal_delta；独立150创新、174 SOURCE残差、旋转和079夹爪回放继续逐步核对。验证真实endpoint、标签、图像、任务稳定和状态误差。先完成准入测试与18 CLI dry-run；所有采集和checker由owner实际wait/reap，完整36stage+owner及来源closure。

资格仍为18/18任务稳定且每条位置RMSE<=.01m、SO3 RMSE<=.05rad。机械验证完成与性能资格分开记录，失败照实保留。尚未执行本计划；通过前不得启动WAMdev6或fresh24。通过后仍须按既定留出新初态门槛完成世界模型参考的闭环验证，不能以oracle成功代替总目标。
