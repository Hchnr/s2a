# 原生目标精度与状态更新响应的停止点命令规划（2026-10-28，CPU）

actual377、cursor9 的 H1/50ms 有限搜索及独立复核均已完整关闭，但没有找到满足原条件的命令。281 个实际评估候选中，87 个满足动作/关节条件，完整位姿 witness 为 0；最终选中候选预测位置误差 0.00963287249 m、旋转误差 0.07346693155 rad，旋转仍超过原 0.05 rad。`complete=true/passed=true` 表示数值执行和核验完成，`witness_found=false/control_qualified=false`；没有派发新 action，实际仍只到 cursor8，task/grasp 没有新增成功证据。

本报告关闭[执行前计划](../plans/v0.0.1/state_updated_stop_control_20261028.md)所登记的停止点数值阶段。已执行计划、源码和产物保持冻结，其中计划末尾“未执行”是执行前快照；当前结果以本报告及下述正式产物为准。前序[响应比较](state_updated_joint_response_20261028.md)和[原生源码合同审计](state_updated_joint_response_source_audit_20261028.md)保留原声明，不能将旧双精目标实验更名为本次原生目标结果。

固定输入来自 H8 原生分支实际 endpoint377，以及同一分支已完成的 step368..376 九周期、225 个内环标签。目标是此前已经生成的 cursor9 TCP，上一命令和 warm 分别来自实际派发记录和旧停止点候选；独立复核的当前 q/v、上一命令、warm、目标、初始关节、kp/kd、关节/力矩限制差异均为 0，`site_from_tcp` 最大差异 1.33e-15。producer 的逐周期状态初始化、先预测后读取标签及 377 绑定见 `scripts/probe_tracking_v5_state_updated_stop_control.py:169`、`:193`、`:218`；checker 独立重建输入和历史见 `scripts/check_tracking_v5_state_updated_stop_control.py:272`、`:423`。

每个候选从当前机器人及两 finger 的 q/v 初始化，按 25×2ms 数值步更新 M/J、FK、bias，并使用隐式阻尼 Euler 自由响应。两 finger 的 q/v 固定在周期起点，外部对象保持 XML 默认状态；接触、关节限位约束、frictionloss 和未建模自由度的加速度耦合仍不在该自由模型中。checker 每次 `mj_resetData`、写入上述机器人状态后以 `mj_forward` 独立查询；没有 `mj_step`、新环境控制轨迹或未来对象输入（`scripts/check_tracking_v5_state_updated_stop_control.py:299`）。

新模型重新预测九个已完成周期，并在每个完整预测之后才消费该周期实际内环标签，逐关节/内步取历史绝对误差最大值，再沿内步作单调累积。最终最大经验保护量为 0.00011565995042106536 rad，profile 复算最大差异 2.78e-17；`future_guarantee=false`。这是当前已完成历史上的经验保护，不是实际未来误差上界。精确零旋转命令使用 step376 起点和实际命令重建的历史方向 goal；非零命令正常重置方向，位置每次重置。九个历史命令均非零，因此该 latch 不依赖更早未知状态（producer `:233`；checker `:333`、`:478`）。

硬条件保持 arm 命令 box、相对上一实际命令 rate 0.25、全部 25 内环关节裕量 0.05 rad 加上述自身历史 profile、终点 `q + 0.1v` 的 0.05 rad 投影裕量，以及 TCP 位置 0.01 m/旋转 0.05 rad。H1 不添加旧 H8 终端方向约束，与已登记的 no-terminal 比较一致；这不认证完整未来安全，也不改变原实际 guard、18 条正确参考、完整 12 条 WAM 开发和最终 24 条新初态门槛。

搜索使用六维 arm 命令加一个辅助违约标量的 SLSQP，单一起点，最多 600 个唯一响应/60 次迭代，无随机起点或重试。所有单边差分都使用归一化命令步长 1e-4，朝 box/rate 交集内部取点，差分和线搜索计入同一预算。warm 仅裁掉第一维 9.71e-16 的舍入越界，原超过 1e-12 拒绝规则不变。新模型下初始安全最小余量为 0.00287422040 rad，因此无需安全恢复阶段；实际仅执行 pose 阶段 32 次迭代，281 个响应，记录耗时 4.01193495 s。该耗时远高于 50ms 控制周期，未取得实时资格。

| 候选（trace 从 0 编号） | TCP 位置误差 / m | TCP 旋转误差 / rad | inner/braking 最小余量 / rad | 动作/关节合规 | 完整合规 |
| --- | ---: | ---: | ---: | --- | --- |
| 初始 warm（0） | 0.00971482658 | 0.07407560439 | 0.00287422040 | 是 | 否 |
| selected（115） | 0.00963287249 | 0.07346693155 | 7.85861820e-9 | 是 | 否 |
| closest_safety_attempt（10） | 0.00959344015 | 0.07347017097 | 4.12282184e-6 | 是 | 否 |

selected 的最紧关节项是 q7 投影制动上界，命令 box/rate 也有活跃边界；其完整内环上界余量为 0.01935613595 rad。最终选择遍历全部真实评估的安全候选，按已登记的 pose 超限排序，并不直接采用 SLSQP 最后一个点。selected 的制动余量低于优化器额外采用的 1e-8 数值缓冲，但通过原关节判断；旋转仍明确失败。SLSQP 返回 `success=true` 仅说明其带辅助违约量的优化终止，不是 pose 达标证明。独立 checker 核对全 trace 的 bounds、每项约束、排序、候选身份和实际迭代/响应预算，不使用 solver success 认证（`src/s2a/nonlinear_single_period_plan.py:98`、`:134`；`scripts/check_tracking_v5_state_updated_stop_control.py:648`）。有限局部搜索无 witness 不能证明全局或物理不可达。

原生 `quat2mat` 实际启用 Numba JIT，`axisangle2quat` 保持 Python。JIT 的 float32 归约经 SciPy BLAS `sdot`，且归一化的标量提升/回写舍入与直接执行 Python AST 不完全一致；NumPy 1.26/2.x 也有标量提升区别。新实现显式复现已固定原生运行时的这些步骤，没有把方向矩阵投影回 SO(3)。独立 checker 从原生源码 AST 提取 `quat2mat` 并实际 `numba.njit(cache=False)` 编译，而不是只运行 Python 副本；未导入新的 response/goal/query/history/constraint/planner 实现。

本次正式复核的 290 个响应（九历史加 281 trace）中，251 次 JIT 与纯 Python 的方向增量矩阵有非零元素差异，最大 2.9802322387695312e-8；producer 与真实 JIT 重建 goal 的最大差异为 0。旧报告中的纯 Python AST/双精目标差异仅属于旧实验，不能充当原生 JIT 等价证据。上轮实现阶段记录的 31 项 CPU 组件检查、9 个历史命令加 seed42 随机 1000 和 zero/tiny/pi 共 1012 个原生 JIT 精确比较予以保留；本轮文档核对没有重跑这些检查。1012 记录见 `src/s2a/native_goal_joint_response.py:17`，不外推为其他 BLAS/编译器/运行时或所有 OSC 运算的逐位保证。

独立 checker 实际运行于 MuJoCo 2.3.7、NumPy 1.26.4、Numba 0.68.0，重算全部 7250 数值内步、7541 次 `mj_forward` 查询，使用 225 个已有实际内环标签，并完整核对 initial/selected/closest 三个不同候选的已归档响应。其统计与 `(9 + 281) × 25 = 7250` 一致（`scripts/check_tracking_v5_state_updated_stop_control.py:704`）。

| 复算项 | 最大绝对差异 |
| --- | ---: |
| 关节位置 / rad | 2.7755575615628914e-17 |
| 关节速度 / rad/s | 1.7763568394002505e-15 |
| requested/applied torque / Nm | 5.933031843596837e-13 |
| goal | 0 |
| 端点 pose | 2.220446049250313e-16 |
| 端点 twist | 1.4988010832439613e-15 |
| trace TCP error | 8.604228440844963e-16 |
| trace safety minimum | 4.440892098500626e-16 |
| 完整候选归一化平方误差 | 7.105427357601002e-15 |

全部满足原 1e-10 状态/几何/指标容差及 1e-8 力矩容差。source refs、九周期因果递推、当前状态和 latch、全部候选判定及选择均独立复核通过，不能据此宣称自由模型等同真实物理。

正式产物和 SHA256 如下；路径均相对本仓库。

| 产物 | SHA256 |
| --- | --- |
| [producer verified](../outputs/tracking_v5_state_updated_stop_control_20261028/verified.json) | `b106b6dc7afe20f0c2d080b3d60c5b98331eca677c5b6cf0a14ab8509c1d8f43` |
| [producer registration](../outputs/tracking_v5_state_updated_stop_control_20261028/registration.json) | `bcfbcd23fff8fc5a9d0535af24ac3c3db16ad95a11e55fc454326dee64fea7c0` |
| [producer owner result](../outputs/tracking_v5_state_updated_stop_control_execution_20261028/result.json) | `c7ef4fc8e7c76e3735088fe6f11e5bc2b5d6369b3a5a395b900826baa658aee9` |
| [checker verified](../outputs/tracking_v5_state_updated_stop_control_check_20261028/verified.json) | `d09c7f9cf9b6685b338dd11b72abaa1f243d9610afcfaf973b2121b9d8ed17ef` |
| [checker registration](../outputs/tracking_v5_state_updated_stop_control_check_20261028/registration.json) | `e18036c2d1e89d6639d051dedaf69a9ffe24946896668dfab85707107983af9f` |
| [checker owner result](../outputs/tracking_v5_state_updated_stop_control_check_execution_20261028/result.json) | `0e3479c2ce2e28d6c77caa86559b88e3e9b9ac1b853be6af92b0bf41beb6aca7` |

两个 owner result 均 `complete=true/passed=true`。各自绑定的 `capture/launch.json`、`capture/exit.json`、`capture/ownership/release.json` 和结果/注册引用 SHA 均核对一致：producer child PID2958646/start_ticks612540871，checker child PID2963860/start_ticks612556501；两者 `actual_wait=true/returncode=0`，`reaped=true/returncode=0`，`signal_sent=false/original_error=null`，没有自动重试。两份 launch 的 `CUDA_VISIBLE_DEVICES` 与 `LD_LIBRARY_PATH` 都为空。

本次只读文档核查重新计算两个 owner 注册中各 1751 个来源文件的 SHA，全部匹配；checker 完成时登记的 52 个来源引用也全部匹配。该链包含原生 XML/控制器/目标变换源码、当前实现、已执行计划、原 H8 状态及既有响应/原生独立检查的退出凭据；没有覆盖旧来源来使检查通过。

本阶段新增物理周期、世界模型调用、fresh case 均为 0。下一步从更早的实际368状态建立状态更新多周期规划，使关键 cursor9 在提前规划时已纳入目标条件；使用当时已经可用的参考和已完成历史，不能借用本次较晚九周期误差 profile 或未来真实状态作在线输入。需另行冻结有限预算、因果残差策略及首点/未来约束，在数值候选独立通过后才能登记 CPU 原生实际推进和独立重放。保留每步实际 TCP、25 内环关节裕量及完整任务门槛；继续禁止 GPU，当前结果不改变总目标未完成的判断。
