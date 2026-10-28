# 提前预防限位：完整固定参考回放对照

2026-10-27。总目标未完成。两个案例各baseline/preventive四分支共512物理周期、独立状态恢复检查均完成。新方案仍无控制资格。

## 实现与已登记参数

`src/s2a/preventive_joint_inverse.py` 在原加速H8动作反解基础上加入全部预测关节的内侧余量平方hinge，margin=.05rad，weight=1000。分别报告跟踪目标与余量目标，不把新增代价下降当成跟踪改善。梯度通过已验证的关节动作Jacobian计算。

首条实际执行动作的模型位置、SO(3)限额分别为max(seed误差, .01m/.05rad)，允许原声明容差内的预防性调整；若seed已超容差，维持其误差上限，不声称能强行满足绝对容差。原动作范围、每维.25变化率、60迭代/160响应、固定OSC及夹爪控制器不变。

`src/s2a/preventive_joint_native.py` 在当前或H8自由预测关节距边界小于.05rad时优化，否则保持原命令。每周期读取实际q/qdot，只执行首动作；不使用raw/terminal标签门控，不注入另一模型的innovation。采用单严格面闭式投影及原多面QP回退，仍是冻结M/J与理想限位模型。

这些margin/weight是旧开发轨迹提示的探索值，没有在fresh上选择，也没有搜索后宣称最佳。

## 仿真对照与范围

统一从原第72步状态开始。每分支真实复现72步原命令前缀，然后使用查询9前8步、查询10前8步、查询11完整40步，共56步，终点128。baseline前16步重算命令与原日志在1e-8内一致。两算法按相同时间表切换同一保存参考和原anchor。

只有当前chunk可进入规划，不提前使用未来chunk；这仍是旧世界模型参考的反事实回放，不是基于候选自身新图像生成的世界模型闭环。worker继承的结果scope字符串含single chunk历史用语，实际参考范围以登记的reference_schedule、逐行query_index/reference_index及最终verifier为准。

| 案例 / 算法 | 56步位置 RMSE mm | 旋转 RMSE rad | 首次q7越界端点 | q7越界周期数 | 抓取接触步 / 任务成功步 |
|---|---:|---:|---:|---:|---:|
| raw baseline | 51.0314 | .178797 | 82 | 47 | 0 / 0 |
| raw preventive | 37.0309 | .178184 | 87 | 42 | 0 / 0 |
| terminal058 baseline | 10.2800 | .036863 | 无 | 0 | 0 / 0 |
| terminal058 preventive | 10.2800 | .036863 | 无 | 0 | 0 / 0 |

raw接受45次模型改进，首次越界推迟5步（.25s），最小q7余量从-.00400648变为-.00320815rad，但仍然越界、仍无抓取或任务成功。这里只比较实际qpos与XML范围，不据此推断限位反力。

为避免不同窗口造成误导，单独比较最后40步（查询11同参考）：raw baseline为60.2437mm/.210715rad；preventive为43.7069mm/.209576rad。此前从88步才干预的H8结果为44.0347mm/.210453rad，提前版本只进一步略有改善。前16步raw位置6.439→4.877mm，旋转.029783→.036294rad，出现姿态退化；不能声称每项误差都改善。

terminal058两方案全部56步实际命令、sim_state/hash、原生goal、夹爪、对象与任务字段精确一致。它的查询窗口与旧query3起点实验不同，不能把当前10.28mm与旧.784mm跨窗口直接比较。

raw预防适配器平均2.44186s/周期，terminal旁路.006799s。baseline日志中的compute_seconds=0表示没有候选适配器，不是原算法完整运行耗时。raw仍不满足.05s实时周期。

## 验证与追溯

- CPU相关5文件：16 passed in9.28s，实际exit0。新增覆盖余量平方hinge梯度的独立中心差分，以及允许首步在声明容差内偏离精确seed来降低余量风险，同时满足动作变化率。
- 第72步模型探针 `outputs/preventive_joint_probe_20261027/verified.json`，SHA256 `54eb26c2467563e1b99b13e128cc490bfc7c23fe2015e67b45cd1b67d88620d6`。raw余量代价.014692→.004616，仅模型诊断；无风险terminal的强制探针对照不能替代实际native旁路。
- 执行 `scripts/execute_early_preventive.py`，统一会话71671实际exit0；4个子进程均由durable_stage_supervisor保存并核验actual wait0/reaped收据，无自动重试。
- 独立 `scripts/verify_early_preventive.py` 实际exit0：复核来源、512周期、配对初态、参考切换、完整动作计划/变化率、首步模型限额及分解目标、原生goal、状态SHA、恢复后状态不变及真实TCP误差。
- 独立核验 `outputs/early_preventive_verified_20261027/verified.json`，SHA256 `3d0b4292f21bcb80c436bb797c98d3ecc9cadc4ab6244a6155e8a97094b4dd56`。
- 分段误差及terminal全物理字段对照 `outputs/early_preventive_verified_20261027/segments.json`，SHA256 `405129201d9c2c97bf861b98a744c5657df4411bd2e5b80514c04f49f580cbe8`。
- 新增实现/worker/owner/verifier/probe及测试Ruff check和format check通过。没有全仓pytest或控制资格声明。

## 结论与后续要求

预防项能够改变接近边界时的行为，但当前软余量代价不能确保避限。提前干预也未消除约4.4cm后段误差，不能把问题归因于晚6步干预就视为解决。

下一项应区分“代价允许牺牲余量”和“在当前接口/首步TCP限额下没有可行余量计划”：以明确的预测关节余量约束检查局部可行性，保留可行/不可行/数值未收敛的区别；不能只不断加权并将较低模型目标当成成功。该局部模型也不能证明真实系统全局不可达，任何新方案仍须真实闭环对照。

本次世界模型调用和fresh案例均0。原正确参考18/18、新候选WAM开发及最终留出24条稳定任务门槛保持，均未由本次局部实验满足。
