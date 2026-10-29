# 085 单周期arm响应残差：诊断与固定因果补偿

084三条消融完整关闭（566步、独立审查/实际退出通过）：native夹爪修复原case10/12，但case03仍失败。085先审查083闭环日志中名义PD响应的残差是否具有可预测的时间结构，不以state误差或命令平滑替代任务结果。

只读审查scripts/audit_arm_residual_persistence.py已完成同18条，输出outputs/arm_residual_persistence_20261025/result.json。逐周期从实际命令仿射解码delta、实际kp/kd的exp(A*dt)重算名义位移和下一线速度，名义线速度与083日志独立核对。位置残差=实际下一位置差−名义位移；速度残差=实际下一线速度−名义速度。用周期t开始已观察到的t−1残差预测t残差；相同t=1..n−2支持下比较固定lag1和零预测，各episode独立。18/18条两通道均改善，case03位置1.561→0.693mm、速度0.04998→0.01939m/s。无fit、物理、模型或fresh；该回顾性预测并不等价于执行补偿后仍然有效。

登记单一候选：保持083名义两状态目标、实际OSC增益、20Hz、旋转0.5/大角保护、命令范围和rate5以及081夹爪策略。周期t用上次实际发送命令的未补偿名义响应与当前实测位置/速度重建上一周期残差，位置/速度系数均固定1，无滤波/累计积分/可调增益。反解时分别从当前desired position increment和desired linear velocity中减去这两个残差，再做原dt²两状态最小二乘和约束投影。预测记录区分名义响应、残差及补偿响应；物理结果仍只读真实状态。

第一周期残差0；每episode/reset清空残差、待观测命令和rate history；正常chunk重规划保留物理历史。只允许当前measured source pose/twist、上一实际发送命令及既有参考，不读未来物理、对象状态、native action或任务谓词。新command的残差必须等下一实际测量后才能形成；不能用本周期事后结果求本周期命令。

先验证状态因果性、实际发送命令（含限幅/rate）作为历史、reset隔离、几何/两状态单位、独立机械回放和完整控制范围。实现/真实解释器dry及084 closure准入完成后，另一个新namespace运行原18条4907步正确参考，原18/18任务及position≤0.01m/rotation≤0.05rad门槛不变，开度误差如实报告。0新模型/fit/fresh；失败不调整本登记参数，不自动重试。通过前不得进入WAM开发或最终新初态。

当前审查已完成；补偿控制器、独立回放、采集/核验/执行入口及物理均未实现/执行。总目标未完成。

实现补充：src/s2a/causal_arm_residual.py已实现固定lag1因果补偿，上一实际受约束命令的未补偿名义响应作为历史；第一周期为零，reset隔离待观测命令/残差/rate状态，无跨episode累计。4项实际CPU检查通过：首周期与083相同、饱和/rate后已发送命令的已知扰动重建与两状态补偿一致、reset隔离、非法观测不推进历史；Ruff通过。当前仅纯转换器实现，完整控制接入/独立机械回放/执行入口/新物理仍未完成，不能推断性能收益。

完整接入补充：causal_residual_loaded_simulation.py保留083 arm参考/坐标/夹爪/时钟/端点逻辑，仅替换因果arm转换并增加nominal预测和残差日志；独立causal_arm_independent_replay.py使用闭式响应/2×1最小二乘复算历史、名义与补偿预测；完整AST对照证明其余控制和机械审查不变。新增独立回放3项、范围2项，加原4项，共9项检查通过。collect/verify/run/close_causal_residual_loaded_control.py及显式config已实现，首条实际3.11 CLI dry通过。执行owner要求083全18条及084全3条实际关闭、同18profiles/4907步、固定lag1系数策略；全部准入后才允许另namespace启动，性能未证明。
