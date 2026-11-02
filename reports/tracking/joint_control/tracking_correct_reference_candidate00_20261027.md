# 首条正确参考候选与独立原生动作回放

## 结果与证据范围

原 case00 候选运行完整164周期，独立检查器另外从相同初态重新执行164个日志 action。全部165端点与记录一致（最大仿真状态差3.3816127675834196e-10，容差1e-8）。独立检查同时重建每8步一次的40步参考编码/解码，检查当前实测源位姿、参考索引、动作边界、arm变化率0.25、夹爪命令、原生位置/旋转目标、原生任务谓词和原位置/SO(3)指标。

独立位置RMSE 0.000533392938581003 m，旋转RMSE 0.0009753354238589933 rad；末尾连续成功93周期。原 stable40、位置0.01 m、旋转0.05 rad均通过。164周期全部走 unchanged_free_feasible 分支，因此仅证明此原案例保持基线，不证明限位优化分支的收益或鲁棒性。

候选与独立回放均 actual_wait0、reaped=true；两个执行owner均实际退出。本次新增物理328周期、世界模型请求0、新初态0。三个最终脚本静态检查通过；不宣称全仓库静态检查或全部算法验证通过。

## 初始化失败与修复

原检查器在创建环境前导入 robosuite，提前选择EGL，在空CUDA配置下初始化失败。v2仅改变模块导入次序，仍在环境创建前导入，继续失败。两次均没有执行回放物理周期，失败owner、exit/release和 verified.json保留。v3将导入放到 make_environment 完成OSMesa配置之后，完成实际回放。所有已登记源码保持原字节，修正使用独立版本文件；候选数据未重新采集。v2存在导入排序静态检查失败，记录保留，不能计为通过。

## 权威文件

- 候选：`outputs/tracking_correct_reference_20261027/case00_candidate/{result,steps,chunks}.json`、`actual_trajectory.npz`
- 原执行及失败检查：`outputs/tracking_correct_reference_candidate00_20261027/`
- 第二次失败：`outputs/tracking_correct_reference_candidate00_replay_v2_20261027/`
- 成功回放owner：`outputs/tracking_correct_reference_candidate00_replay_v3_20261027/`
- 独立结果：`outputs/tracking_correct_reference_check_v3_20261027/case00/verified.json`
- 成功检查入口：`scripts/check_tracking_correct_reference_v3.py`

## 未完成项

检查器证明已记录action在固定原生物理接口下的执行与指标，不独立证明优化器最优性、所有规划输入的因果性、动态模型约束证书或图像/标签全量资格。后续还需补齐适用范围内检查，并运行其余原17案例，保持原18/18资格门槛；再进行WAM开发和最终24条新初态验证。当前总目标未完成，不将单条结果计作完整候选资格。
