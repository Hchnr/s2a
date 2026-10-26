# SQP六维联合逆物理回归

SQP物理回归前3条共601周期已通过独立任务与位姿门槛，尚无完整18条性能结论。232完整控制基线仍为17/18稳定并综合达标；SQP正在按246修复后的入口继续固定18条物理回归。原18/18资格、237世界模型开发及最终fresh24门槛保持，总目标未完成。

## 已完成准入

- `outputs/full_pose_sqp_all_records_20261026/independent_verified.json`：原18条4907周期全量记录、各可行reset/latch模式的独立ODE/有限差分梯度/KKT、native goal和双历史通过，最大独立驻点残差7.561676843492389e-7，小于原2e-6门槛；完整父来源图开始和结束均核验。
- `outputs/full_pose_sqp_control_log_20261026/independent_verified.json`：164周期JSON往返、164次SOURCE commit、20项损坏拒绝，最大独立驻点残差7.457963500240304e-7。
- 两个worker均actual wait0/reap、无信号且同身份退出；对应matrix/log receipt audit和supervision execution_result完整。这些是记录/重建控制器历史的资格，不能当成改变命令后的物理闭环结果。
- 246完整真实来源预检通过：1426项runtime、7646项父来源和7713条原声明映射。3项路径回归及修复四文件Ruff检查通过，旧注册输出配置SHA与当前配置一致：`a4b94475a6906a1aa05526b992fda80434aa2e62beabd49d81f64af1a1e6ad0f`。

## 启动修复与执行边界

245注册actual wait0，但首次control_launch在创建物理ROOT、测试与仿真前，因历史source_sha256含合法相对路径而触发绝对路径断言，actual wait1/reap且无信号。旧记录和冻结代码保留，新repair不重跑数学采集或注册。

`scripts/full_pose_execution_sources_resolved.py`以REPO解析相对声明，规范化后合并并拒绝SHA冲突，保留原声明/路径/SHA映射；实际来源和性能门槛不变。`scripts/repair_full_coupled_sqp_admission.py`在独立`outputs/full_coupled_sqp_admission_repair_20261026`中监督`run_full_coupled_sqp_control_resolved.py`，成功后安排原独立closure。

物理namespace为`outputs/full_coupled_sqp_control_20261026`；固定原18 profiles/4907周期、seed42、079夹爪、173最终SOURCE commit与150独立历史不变。新owner先运行40个测试模块、18个真实CLI dry及checker导入；通过后才执行36个采集/独立核验阶段。CPU/OSMesa显式空CUDA/LD，三项线程限制仍8，无模型调用或fresh样本。

每条必须任务稳定、位置RMSE≤0.01m、SO3 RMSE≤0.05rad；36stage和owner actual wait0/reap、来源关闭及18/18全部性能达标后，才允许237开发。求解和推理耗时单列，当前不声明实时控制性能。

246监督启动已完成ready/source准入，`outputs/full_coupled_sqp_admission_repair_20261026/supervisor/launcher_result.json`为complete/passed；最近/proc核实监督PID878317/start_ticks594005721与新准入worker879235/start_ticks594012234均同身份存活。新worker正执行完整来源/测试准入，物理namespace尚未创建，不能将入口启动当作物理性能通过。

记录日志求解耗时审查：164次solve_pose调用中位数34.038ms、线性插值P95 39.263ms、最大145.784ms，2次超过50ms。原生8线程且存在并行CPU工作；不含整个状态读取/控制周期，也不是物理实时资格。来源SHA与完整统计保存`outputs/full_pose_sqp_recorded_solver_timing_20261026.json`，后续物理日志仍需单列实际求解开销。


## 物理执行首批结果

246新准入完成：143项测试（53.31秒）、18条真实CLI dry与checker导入均实际return0；control_launch actual wait0/reap，repair execution_result complete。物理owner898994/start_ticks594206716、独立closure监督899090/start_ticks594207420经/proc同身份核实存活。

首3条完整独立机械/标签/图像审查及6个实际阶段wait0/reap核实通过，配对证据`outputs/full_coupled_sqp_prefix_03_20261026.json`保存同一原profiles及旧232比较：

| case | SQP位置RMSE | SQP SO3 RMSE | 旧232位置RMSE | 旧232 SO3 RMSE | SQP稳定/达标 |
| --- | ---: | ---: | ---: | ---: | --- |
| 00 | 9.26798mm | 0.0497706rad | 1.10798mm | 0.00378261rad | 是/是 |
| 01 | 0.27980mm | 0.00103539rad | 0.29188mm | 0.00959077rad | 是/是 |
| 02 | 0.40928mm | 0.00066204rad | 0.42076mm | 0.00362711rad | 是/是 |

case00跟踪明显退化且接近原上限，末尾仍连续80步成功；case01/02旋转误差下降。不能由首3条达标宣称整体改善或18/18资格。case03已有171实测端点，但在完整独立证明和实际阶段退出核实前不计为通过；冻结配置继续执行，0新模型/fresh。
