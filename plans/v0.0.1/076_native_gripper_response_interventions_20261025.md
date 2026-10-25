# 076 训练初态下的夹爪命令干预与因果响应采集

074正确参考14/18失败、075前向辨识未过原门槛；状态审计发现训练缺少“目标motor饱和但开度仍小且缓慢打开”的响应，瞬时速度可在线观测但并非已证明的唯一缺失因素。本阶段先采实际前向数据，不继续调同一6条内部留出。

固定取原24fit中的首个每任务demo（bowl/demo8、wine/demo1、cream/demo2），与原早期pilot相同，不碰development或fresh。每条从原注册初态运行原native命令前缀，不把测量opening重写为motor cache。命令anchor仅由原日志决定：出现过正闭合命令后，第一次负打开命令的周期；在执行之前生成确切配置与预算，不根据新响应或成功挑anchor。

每个demo四分支各自独立重建环境并重放完整相同前缀：原native后24步基线；前8步仅夹爪强制-1打开、0保持或+1闭合，随后16步恢复原native命令。臂命令始终按同一原日志，固定20Hz及native API/25子步，不改控制器增益或物理参数。共12条，确切预算为每条anchor+24步，生成配置时冻结。各分支都从原初态重放，禁止未验证的sim/controller/cache隐状态恢复。

所有实际端点记录source/TCP、opening、两个finger qpos和当前qvel、完整sim state、真实motor cache、实际发送命令、物理time、控制器合同及源码/数据SHA。观测qvel必须与sim当前qvel索引一致；记录对象/接触只能作为离线诊断，不进入拟合输入。基线先与原native前缀逐周期验证当前资产下的状态/缓存/时间一致性；若失败保留，不启动干预，不通过放宽容差冒充同一条件。

先实现并核验基线采集、独立端点/命令motor累计审查及实际3.11 dry，再启动固定批次。CPU/OSMesa空CUDA/LD，seed42。所有stage由owner实际wait/reap，另由等待包装进程持有owner并保存其实际wait退出码，避免074根owner退出凭据缺口。不得自动重试或覆盖原输出。

本阶段只采集/验证数据，不自动fit或接入控制；下一阶段须先登记如何划分新干预序列及使用哪些训练轨迹，隔离保留新的响应验证条件，再评估单步和多步state误差。原074/075失败及WAM开发/fresh门槛不变，总目标未完成。

配置已生成：configs/native_gripper_response_interventions_20261025.json，三任务anchor分别78/157/85周期，12条总1568物理步（尚未执行）。来源为原固定fit轨迹及原sent_commands；每个anchor按上述纯命令规则计算，前缀+24均在已有原命令长度内。实现与独立准入仍待完成，未启动采集。

执行前合同细化（尚未采集）：复用原settled native采集的SettledGoalEnvironment，在native_steps后按同一既有协议锁定当时实测controller SITE位姿，不能省略该封装或把重复动作当作相同的末段控制条件。各分支都复用此协议。baseline原记录sim_states/source_pose/opening/gripper_qpos逐周期绝对容差1e-10、motor cache1e-12、time间隔0.05容差1e-10；这些容差在首次执行前固定。独立审查从实际model.xml重建qpos/qvel索引核对观测，使用原actual endpoint标签检查器；此为响应数据采集，不要求图像或任务成功。六项命令干预测试已通过，实际3.11 case0 dry通过，完整12case dry和源码/输入冻结由launch准入统一执行。
