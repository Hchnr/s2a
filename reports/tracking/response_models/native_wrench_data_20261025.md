# 103 原30轨迹当前wrench数据：完整资格通过

最新完整结果：原30条/24fit/6holdout、6180显式控制周期/6210当前端点均独立合格，60阶段与owner实际wait0/reap及来源不变通过，原六102 wrench共同前缀匹配。关闭证据 outputs/native_wrench_data_20261025/closure_verified.json。仅数据资格，无新控制/任务收益；104拟合已完成但21/24预测、22/24响应，离线资格失败。以下保留运行期间快照。


沿用087原30轨迹、24fit/6holdout及原初态/seed42/全部native与80周期settled命令，完整预算6180控制周期、6210端点。每端点直接读取102已验证的机器人传感器getter和sensordata，保存local6、world-at-control-SITE6、sensor pose、control SITE位置、时间和身份；读取在原endpoint刷新后、新命令前，不做新的forward/step，不从旧snapshot重算力冒充测量。

原始native数组无source_twist，参考绑定087合格cache的x_site_twist加末行y_twist构成N+1端点；各绑定均核对SHA。新独立检查器使用XML和完整state重建FT传感器site位姿，再核对世界力/力矩及参考点平移；完整sim_state/SITE pose/twist仍须与原native一致。保留24fit/6holdout原身份，不拟合模型。

4项既有核心测试及30真实CLI dry-run准入通过，持久owner已启动。当前核实第0条完整164控制周期通过独立检查（含80终端周期），第1条worker与owner PID/start_ticks匹配且存活，未见执行错误；此为部分运行快照，不是30条完整资格。

完整关闭入口scripts/close_native_wrench_data.py已实现，要求30条/6210端点、60阶段与owner实际wait/reap和来源不变，并把102原六baseline的全部共同wrench前缀与本次native数据逐项匹配；尚未执行全量关闭。生成namespace outputs/native_wrench_data_20261025，未覆盖旧产物，初始化内部步骤另列。世界模型/fresh/fit均0，原18/18正确参考、开发和最终24条新初态验收保持，总目标未完成。

接续核验快照：13/30条独立审查、2491控制周期，owner及group13 worker存活、身份匹配，未见执行错误。等待期间仅新增104当前实测wrench条件核心/独立读取器及7项检查，不修改103冻结源码或提前拟合。
