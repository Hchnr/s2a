# 085 因果arm残差可预测性：只读审查通过，控制未执行

083已完整关闭的18条日志中，以上一周期已观测到的残差预测下一周期名义PD模型误差，位置和线速度均在18/18条优于零预测。所有比较使用相同t=1..n−2支持，每episode隔离，实际发送命令解码、kp/kd指数响应及日志名义速度一致性均核验；输入SHA前后不变。

| 原case | 位置零预测/lag1 RMSE（mm） | 速度零预测/lag1 RMSE（m/s） |
| --- | --- | --- |
| 03 | 1.561 / 0.693 | 0.04998 / 0.01939 |
| 10 | 2.302 / 0.586 | 0.07348 / 0.01759 |
| 12 | 0.809 / 0.349 | 0.02465 / 0.01130 |

依据：scripts/audit_arm_residual_persistence.py及outputs/arm_residual_persistence_20261025/result.json。Ruff检查通过，0物理/fit/模型/fresh。日志残差持续性不能证明施加补偿后的任务收益；[085计划](../plans/v0.0.1/085_causal_arm_residual_20261025.md)固定一个无调参的单周期因果补偿作为下一候选，尚未实现或执行。原WAM开发和最终留出门槛保持，总目标未完成。

实现补充：src/s2a/causal_arm_residual.py已实现固定lag1因果补偿，上一实际受约束命令的未补偿名义响应作为历史；第一周期为零，reset隔离待观测命令/残差/rate状态，无跨episode累计。4项实际CPU检查通过：首周期与083相同、饱和/rate后已发送命令的已知扰动重建与两状态补偿一致、reset隔离、非法观测不推进历史；Ruff通过。当前仅纯转换器实现，完整控制接入/独立机械回放/执行入口/新物理仍未完成，不能推断性能收益。

完整接入补充：causal_residual_loaded_simulation.py保留083 arm参考/坐标/夹爪/时钟/端点逻辑，仅替换因果arm转换并增加nominal预测和残差日志；独立causal_arm_independent_replay.py使用闭式响应/2×1最小二乘复算历史、名义与补偿预测；完整AST对照证明其余控制和机械审查不变。新增独立回放3项、范围2项，加原4项，共9项检查通过。collect/verify/run/close_causal_residual_loaded_control.py及显式config已实现，首条实际3.11 CLI dry通过。执行owner要求083全18条及084全3条实际关闭、同18profiles/4907步、固定lag1系数策略；全部准入后才允许另namespace启动，性能未证明。

## 085实际启动

20个准入进程全部通过（60项CPU测试、全部18条真实3.11 CLI dry及独立checker导入）。新namespace outputs/causal_residual_loaded_control_20261025已冻结来源/config/计划，同一组18条4907步；进程身份经/proc核实：

[
  {
    "stage": "owner_wait/launch.json",
    "pid": 3682199,
    "start_ticks": 584150081,
    "live": true
  },
  {
    "stage": "orchestration/case00/launch.json",
    "pid": 3682207,
    "start_ticks": 584150118,
    "live": true
  }
]

CPU/OSMesa显式空CUDA/LD，0模型/fit/fresh。当前尚无完整闭环性能结果，不能由残差预测改善推断任务收益，原WAM/留出验收门槛保持。

## 前两条独立核验通过

085 case00/01均稳定并满足原门槛。位置RMSE分别2.016/0.623mm，相同参考083为3.217/2.008mm；旋转为0.007368/0.009758rad，均在0.05rad内但比083稍大。完整性能结论仍待18条。

新增只读scripts/compare_causal_residual_tracking.py使用两者共同固定source目标位置作中央差分、排除首尾，不以不同实际query anchor生成的速度作为公共真值；新输出outputs/causal_residual_fixed_reference_pairs_20261025/verified_prefix_02.json，case00速度RMSE085/083为0.018907/0.017425m/s（退化），case01为0.007497/0.011673m/s（改善）。旧common085-query-anchor口径的prefix01保留在outputs/causal_residual_tracking_pairs_20261025，属于不同描述口径，不混作同一数值序列。命令变化及速度收益并非一致，不能由位置改善推断任务可靠性。0新物理/模型/fit/fresh。

## case03首次在该控制链恢复稳定

085前三条均通过，case02位置0.816mm/SO3 0.003981rad。随后case03完整独立机械/因果残差/标签/图像审查通过，ever/terminal/stable均True，末尾88步成功，位置1.799mm/SO3 0.009499rad，原门槛通过。083同条失败，084固定083 arm加native夹爪仍失败；085保持081夹爪而增加因果arm残差后首次在该链通过。不能据此推断唯一物理原因或完整泛化。现4/4合格，剩余14条仍在原owner执行；原18/18与WAM开发/最终fresh门槛不变。前4配对保存outputs/causal_residual_fixed_reference_pairs_20261025/verified_prefix_04.json，0额外物理/模型/fit/fresh。

前6条独立核验均达标（6/6）。case04位置RMSE1.591mm/SO3 0.007237rad、case05位置0.555mm/SO3 0.009897rad，任务均稳定；原owner PID3682199/start_ticks584150081再次核实存活。尚有12条未完成，不能将该prefix替代完整18/18或最终新初态验收。

前9条独立核验全部达标（9/9）。case06位置5.404mm/SO3 0.014120rad、case07位置1.440mm/SO3 0.004736rad、case08位置3.168mm/SO3 0.007257rad，均任务稳定。原owner PID3682199/start_ticks584150081仍存活，仍有9条未完成；不据半程结果提升18/18或WAM/fresh资格。

前11条独立核验均通过（11/11）。case09位置RMSE9.325mm/SO3 0.014433rad，虽合格但比083位置4.086mm更差，接近原10mm阈值，不能声称统一改善。case10在原083失败后恢复稳定，末尾84步成功，位置7.736mm/SO3 0.020223rad，原门槛通过；保持081夹爪策略，没有使用084的native夹爪oracle输入。原owner身份持续核实存活，尚有7条未完成，完整18/18/WAM/fresh资格仍未取得。

## case11旋转门槛失败

case11任务稳定、末尾87步成功，但位置RMSE5.909mm/SO3 0.072653rad，超过原0.05rad旋转门槛，因此综合失败，085原18/18资格已不能通过。原083同条SO3 RMSE0.009327rad，构成新增跟踪退化。前12条11条合格，剩余6条继续执行。

只读时间定位保存原输出目录case11_rotation_diagnosis.json：085旋转最大误差0.37555rad在周期173，前0–80周期与083近似，主要退化在后段；周期160–180伴随大平移误差、命令限幅/变化率限制及毫米级响应残差。存在同步现象但尚未隔离唯一根因；不能据任务稳定或原旋转算法未变而免除旋转门槛。0额外物理/fit/模型/fresh。

case12、13均独立核验并达标，前14条13条合格。case11额外只读XML关节range及实际qpos核查：085全程最小硬关节限位余量0.6165rad，旋转峰值周期173后最近余量0.6258rad，未触及硬限位；083全程最小0.6671rad。仅排除该直接硬限位解释，不代表Jacobian条件数、力矩限制或耦合已排除。输入SHA前后核验和逐关节余量保存在case11_joint_margin_diagnosis.json，0额外物理/fit/模型/fresh。

case14独立机械/因果残差/标签/图像审查通过，但任务从未成功，位置RMSE43.662mm、SO3 0.236673rad，均超原门槛；083同条原稳定合格，构成明显新增退化。前15条13条合格（case11旋转失败、case14跟踪与任务失败），不能因case03修复或部分训练参考改善而推广单位lag1补偿。最后3条仍按冻结登记执行，不在本轮修改系数或自动重试。

实际085闭环日志再审查（scripts/audit_causal_residual_closed_loop.py，outputs/causal_residual_on_policy_audit_20261025/verified_prefix_15.json）：15/15条的位置/线速度残差校正预测误差仍低于名义预测；失败case11位置2.961→0.940mm、速度0.09553→0.02842m/s；case14位置4.228→2.291mm、速度0.13523→0.07881m/s。case11/14分别38/83个饱和周期、55/105个rate限制周期。预测改善与控制退化同时存在，说明只验证加性偏差可预测性不足以证明逆控制局部增益/耦合和闭环稳定性。0额外物理/fit/模型/fresh，下一步应查实际命令响应而非据此挑新残差系数。

case15完整独立核验后任务稳定，但位置RMSE12.303mm超10mm门槛（SO3 0.024505rad），综合失败。现前16条13条合格，失败case11/14/15，最后2条尚未完成。基于“预测改善仍控制退化”，另登记[086成对arm响应采集](../plans/v0.0.1/086_paired_arm_response_20261025.md)：只用旧24fit中的demo8/1/2，在close/open两anchor采集baseline和translation±0.1分支，固定42条3339步；无fit/新控制/模型/fresh，必须085完整关闭后才允许物理。当前仅计划/config，未实施或执行。

## 完整关闭

18条4907步、全部逐条独立机械/因果历史/标签/图像审查完成，36个阶段及owner实际wait0/reap、无信号与来源不变均通过。regression_summary.json和closure_verified.json保存原输出目录。任务稳定17/18，综合资格15/18，失败case11/14/15，原18/18门槛失败。不采用直接单位lag1残差补偿进入WAM/fresh；它修复部分任务但引入跟踪和任务退化，预测误差改善不是控制稳定性证据。086成对响应采集仍仅实现组件，尚未执行新物理。总目标未完成。
