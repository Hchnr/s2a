# Seed43正确参考执行资格 — 2026-10-21

**015的三条旧初态oracle仿真全部通过执行资格：3/3原生末40端点稳定成功，位置/SO(3)跟踪误差均通过原10mm/0.05rad门槛。** 使用正确参考，不涉及WAM生成；不能将此结果计入WAM成功率或seed43 WAM响应资格。

[015计划](../plans/v0.0.1/015_seed43_execution_qualification_20261021.md)与[配置](../configs/seed43_execution_qualification_20261021.json)在`a1c55a3`冻结。固定三个旧task/init0、原fixture、v3位置夹爪、无governor、每8物理步重观测、H40/20Hz、每条600步。实验seed为42，实际环境/日志sampling seed为43。root在独立预算下执行3条/1800物理步、0模型请求、0fresh初态；本报告的审计只读保存日志，新增物理步/模型请求均为0。

| 旧初态 | TCP位置RMSE / mm | SO(3) RMSE / rad | 原生成功端点数 | 尾部连续成功端点 | 原门槛 |
| --- | ---: | ---: | ---: | ---: | --- |
| bowl / task0_init0 | 1.838090 | 0.003339 | 531/600 | 531 | 通过 |
| wine / task1_init0 | 1.421041 | 0.006744 | 458/600 | 454 | 通过 |
| cream / task2_init0 | 1.547627 | 0.002595 | 501/600 | 501 | 通过 |

[新只读入口](../scripts/audit_wam_seed43_qualification.py)直接调用已在`1444f33`冻结的seed-aware完整检查器。逐case验证43属于配置、result seed为43，重算全部600步机械/几何、夹爪、原生结果、参考与时钟；每条75个oracle chunk。三个case没有任何WAM request/response文件。补充核对worker参数`--sampling-seed 43`、`--seed 42`、`--source oracle`，manifest中实际runtime seed和result seed均为43；实验seed42与环境/日志seed43没有混用。

随后对007中相同旧fixture、interval8/no-governor/v3的真实seed42日志执行同一完整检查。三个新旧case的所有1800个step字段逐值精确相同；225个chunk除wall-clock `generation_s`外也全部精确相同；完整case返回字段仅路径不同。此比较保留全部控制命令、motor/qpos、开度、目标/实测位姿、控制器goal、时钟、对象与原生成功记录，没有通过修改42日志构造43证据。

389个输入文件SHA前后核验通过。root启动的CPU matrix PID235870/startticks167841386及三个已登记physics worker均已退出，matrix完整、三个worker返回0；CPU CUDA/LD为空。旧执行器及全部既有checker字节保持原样，Ruff/格式检查通过。

可复核产物：

- [完整资格审计](../outputs/wam_stability_20261021/seed43_execution_qualification/qualification_independent_verification.json)：`9cd4cf46260eda2d87e13bd992c5ce871469d4ff2d3f38734129e2a1bd5fbba3`，`complete=true`、`qualified=true`。
- [实际runtime/worker seed补充](../outputs/wam_stability_20261021/seed43_execution_qualification/runtime_seed_metadata_supplement.json)：`ab522e75318da9d708bf8b1d060c539d83c60fd4d9607101f876ce6ec530fe08`。
- [独立启动登记](../outputs/wam_stability_20261021/seed43_execution_qualification/orchestration/root_launch.json)：`04a756adfd1761f76bace8a239821a4c7d8785b26b34d56f661cb379269e9340`。

恢复的是同一组完整物理初态，因此seed43没有引入新的初态分布；新旧轨迹精确相同符合此条件下的确定性执行。这里确认的是正确参考与实际43环境/日志分支，未验证43的当前图像→WAM请求→模型响应链，也不证明WAM在多seed/新初态稳定。014旧42兼容性回归、015正确参考资格和010 WAM开发实验分别保留分母。
