# 较早实际状态的原生目标多周期规划（2026-10-28，CPU）

actual368 的 H12 单次搜索已正常结束，但没有找到满足全部原条件的轨迹。4000 个唯一完整轨迹响应耗尽预算，完成 53 次迭代；producer 记录 406 个动作/关节合规候选、完整 witness 为 0。selected 的最大位置误差 0.02255697719 m、最大旋转误差 0.11277641832 rad，均超过原 0.01 m/0.05 rad。没有派发新 action，`control_qualified=false`，实际仍止于此前分支的 cursor8，未取得 cursor9 实际推进或新任务成功。

当前证据状态：producer及其owner、v2独立checker及其owner均`complete=true/passed=true`，实际wait0/reap0、无信号。v2从头独立复算全部4000轨迹/1200000内步/1248001次MuJoCo查询，确认406个仅动作/关节合规候选、完整witness0，selected/closest索引2224/151及53迭代一致。原独立v1检查只完成500/4000条即因开销被明确停止，actual wait/reap=-15；该失败/停止产物和源码原样保留，未并入v2通过计数。数值复核通过只确认本轮计算与判定一致，不能将无witness提升为控制或任务成功。

本次执行按[事先登记计划](../plans/v0.0.1/state_updated_trajectory_control_20261028.md)，承接[停止点 H1 搜索失败](state_updated_stop_control_20261028.md)，转到更早的 actual368 状态，唯一 H12 窗口覆盖已生成 chunk46 的 cursor0..11，明确包含 cursor9 及其后两点。唯一 warm 是旧同状态 H12 的 `answer.planned_commands`，不追加随机起点、参数扫描或其它初始化；seed42。producer/wrapper/solver/owner 提交为 `ef67eee`，独立 checker 提交为 `d976e84`。

起点仅使用原 `causal_error` 分支的实测368 arm/finger q/v，初始误差 profile 为全零，`completed_periods=0/last_completed_step=null/future_guarantee=false`；没有把较晚377累积的九周期残差倒灌到368。方向 latch 取已完成367的真实 `controller_goal_after`，并用该周期 SITE 和非零实际旋转命令重建核对。每个后续预测周期仅继承自身预测 q/v 和方向 goal，零旋转保持历史 goal、非零旋转重置；无未来实际状态、对象或接触作为策略输入。来源与初始化见 `scripts/probe_tracking_v5_state_updated_trajectory_control.py:73`；跨周期递推见 `src/s2a/native_goal_trajectory.py:51`。

响应使用已固定的原生 JIT goal 精度、状态更新 M/J、25×2ms 自由 OSC 和隐式阻尼 Euler；两 finger q/v 在整个预测窗口固定为当前368状态，外部对象维持 XML 默认状态。接触、限位力、frictionloss 和未建模自由度加速度耦合仍被省略。这是数值预测模型，不是实际控制或安全证明。

全部12个预测端点同时要求 TCP 位置≤0.01 m、旋转≤0.05 rad；全部300个内环关节保持0.05 rad裕量，全部12个端点的 `q + 0.1v` 保持0.05 rad裕量，最后 q7 速度≤0（固定原 `+e7` terminal）。命令 box、相邻周期 rate≤0.25、第一周期相对实际367命令的 rate 都保持。当前 profile 全零，所以该轮历史保护不额外扩大裕量。原 box/rate 1e-12、关节/terminal 1e-10 容差及真实执行 guard 均不变。

优化为72个arm命令变量加非负辅助违约量的 SLSQP，最多4000个唯一完整响应/60迭代、单一起点。初始非线性安全最小余量0.04034866571，已满足条件，因此没有进入最多20迭代的安全恢复阶段，仅执行 pose 阶段。归一化差分步长固定1e-4、`ftol=1e-12`，优化额外内余量1e-8；差分探针遵守box，但允许rate违约且不能作为最终可行候选。目标、约束、差分、线搜索共享精确命令字节 cache；全部4000响应保存，不把12周期预测算成一次实际控制。

本次终止原因为 `response_budget`，不是60迭代上限：53迭代时用满4000响应，`success=false/status=finite_search_without_witness`。累计1200000数值内步、48000预测控制周期，记录搜索耗时650.08665120 s；远超20Hz控制周期，未取得实时资格。trace 中1030个差分/搜索候选存在超过原容差的rate违约，仍计入预算并保留为失败候选；406个动作/关节合规候选均未满足全部12点精度。

| 候选（trace从0编号） | 最大位置误差 / m | 最大旋转误差 / rad | 同时满足两项pose caps的点数 | 归一化最大超限 | 动作/关节合规 | 完整witness |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| initial（0） | 0.03102201148 | 0.23286633465 | 6/12 | 3.65732669 | 是 | 否 |
| selected（2224） | 0.02255697719 | 0.11277641832 | 2/12 | 1.25569772 | 是 | 否 |
| closest_safety_attempt（151） | 0.02839285334 | 0.18403845664 | 6/12 | 2.68076913 | 是 | 否 |

这里“归一化最大超限”为 `max(position/.01, rotation/.05) - 1` 的非负部分。selected 在实际评估且动作/关节合规的候选中按该值、再按归一化平方误差和排序；它改善了最坏超限，却把同时达标的点数由warm的6/12降到2/12，不能报告为全面改善。closest按所有动作/关节组最坏负余量、再按平方误差和排序，与selected目的不同；容差内约1e-15的rate负余量也会影响这个诊断排序。两者都不是完整合规轨迹，更不是已经执行的结果。

selected的全部12点预测误差如下。cursor编号与原已生成参考一致，未改目标时间或跳过关键点。

| cursor | 位置误差 / m | 旋转误差 / rad | 原双caps |
| --- | ---: | ---: | --- |
| 0 | 0.00253920838 | 0.02109681375 | 通过 |
| 1 | 0.00866953790 | 0.04051695195 | 通过 |
| 2 | 0.01394664411 | 0.06865020673 | 失败 |
| 3 | 0.01706002670 | 0.09285967453 | 失败 |
| 4 | 0.02029251129 | 0.10991871807 | 失败 |
| 5 | 0.02253354185 | 0.11277641832 | 失败 |
| 6 | 0.02216283112 | 0.10965879520 | 失败 |
| 7 | 0.02192193839 | 0.08791815028 | 失败 |
| 8 | 0.02255697719 | 0.05585945335 | 失败 |
| 9 | 0.02114467464 | 0.05572895012 | 失败 |
| 10 | 0.02255430502 | 0.07056159835 | 失败 |
| 11 | 0.02255524396 | 0.11276652611 | 失败 |

selected所有约束组的最小余量和活跃数量如下。动作box/rate余量无量纲，inner/braking单位rad，terminal单位rad/s；活跃阈值为1e-8，不能混合这些组的数值大小推导物理瓶颈。全部组在原容差内，无违规索引。

| 约束组 | 最小余量 | 活跃数量 | 最小值位置（cursor/内步与轴） |
| --- | ---: | ---: | --- |
| box lower | 1.11022302e-16 | 1 | cursor11，arm命令第3轴 |
| box upper | 0.21751195390 | 0 | cursor1，arm命令第5轴 |
| rate lower | -1.38777878e-15 | 13 | cursor7，arm命令第3轴 |
| rate upper | -2.77555756e-16 | 14 | cursor2，arm命令第1轴 |
| inner lower | 0.24733028576 | 0 | cursor6结束，q4 |
| inner upper | 0.00577226975 | 0 | cursor11结束，q5 |
| braking lower | 0.23715864677 | 0 | cursor3，q4 |
| braking upper | 2.17121263e-5 | 0 | cursor6，q7 |
| terminal `-q7dot` | 6.65422765e-5 | 0 | 最后cursor11端点 |

最紧非线性安全项为cursor6的q7投影制动上界，最后q7速度为−6.65422765e-5 rad/s；所有300内环中最小关节上界余量为0.00577226975 rad。动作box/rate有活跃条件，但本次失败的最终判据是pose caps；不能仅凭活跃索引宣称某一约束是唯一根因，亦不能用有限局部搜索失败证明物理不可达。

本阶段执行前五个CPU组件文件合计47 passed（5.48s），包括 `test_native_goal_joint_response.py`、`test_nonlinear_joint_response.py`、`test_nonlinear_single_period_plan.py`、`test_native_goal_trajectory.py` 和 `test_nonlinear_trajectory_plan.py`；CUDA/LD均为空。证据为root会话工具session79332的实际pytest输出，临时测试目录为 `outputs/trajectory_component_tests_20261028_tlsHSl/pytest`；没有单独持久stdout产物，本报告不虚构QA日志SHA，也未为写报告重新运行测试。producer及checker源码Ruff/format检查通过；组件检查不构成轨迹性能或控制资格。

已关闭的producer证据如下，路径均相对仓库。

| 产物 | SHA256 |
| --- | --- |
| [producer verified](../outputs/tracking_v5_state_updated_trajectory_control_20261028/verified.json) | `89626cd5884b50de1cc4e63d1a5bf46b966ff59bc6df82f7e2dc2b4ea71d563e` |
| [producer registration](../outputs/tracking_v5_state_updated_trajectory_control_20261028/registration.json) | `0e55eed9927774a9a0543b4aed9e87b7d716a20aaec7d43530d42df49c35f5ae` |
| [producer owner result](../outputs/tracking_v5_state_updated_trajectory_control_execution_20261028/result.json) | `82b813eb1914ae81f516bcac5b7e214fd9d4ea8c1fe4ceff7aef186eea241075` |
| [producer owner registration](../outputs/tracking_v5_state_updated_trajectory_control_execution_20261028/registration.json) | `3c4ad79552b75f16cee60268565e4ceab21ac76e6d6f74cd6cb5d4f0de35fce5` |

producer launch绑定child PID3012108/start_ticks612685767；`actual_wait=true/returncode=0`，release `reaped=true/returncode=0/signal_sent=false/original_error=null`，同一进程身份和结果/注册/launch/exit/release引用SHA核对一致。`CUDA_VISIBLE_DEVICES`、`LD_LIBRARY_PATH`均为空，无自动重试。原producer及原v1 checker两个owner注册各固定1754个来源，包含已执行计划及新旧实现；本次文档核查已重新计算全部来源SHA，均匹配。该来源核对本身不替代下述v2全量独立数值验证。

原独立checker注册 [registration.json](../outputs/tracking_v5_state_updated_trajectory_control_check_20261028/registration.json)，SHA `01b87d56a48d3f3028219d426ec2e6085c80c987df06e2d058c20c3fd59b8fbd`；其owner注册SHA `c65621a0170bae90cf465c4dd73f00b176aa02c1106278d789c5f581c77f46e7`。launch child PID3028290/start_ticks612754824，CUDA/LD均为空。登记范围为独立原生AST+真实Numba JIT、每次resetData/mj_forward、完整4000轨迹/1200000内步的跨周期重算，并核对initial/selected/closest完整rollout、全部原约束、排名、来源及预算。

实际运行观察到300条约5分钟，按该速度无法在原1200秒owner时限内完成全部4000条。root先记录[停止决策](../outputs/tracking_v5_state_updated_trajectory_control_check_stop_20261028/decision.json)，再核对child身份并外部发送SIGTERM；最终日志停在500/4000条、150000数值内步。这是检查执行成本导致的未完成，未出现已归档的数值不一致判定，也不是根据候选结果选择保留通过项。原owner实际退出1，记录 `complete=false/passed=false`，child `actual_wait=true/returncode=-15`、`reaped=true/returncode=-15`；父owner `signal_sent=false` 只说明信号不是父owner发出，外部root发信号事实保留在停止决策中。不存在原checker完整 `verified.json`，不能将500条进度日志提升为全量独立通过。

| 保留的原独立attempt产物 | SHA256 |
| --- | --- |
| [明确停止决策](../outputs/tracking_v5_state_updated_trajectory_control_check_stop_20261028/decision.json) | `d7f5f9391f7d8e64bb87620ee04815f544448229c5bfa7c05a48ca40669bef37` |
| [原checker失败owner result](../outputs/tracking_v5_state_updated_trajectory_control_check_execution_20261028/result.json) | `39d0145d2f0f0c32723774c0309663e6b0bede2f68c03b03bd56b5684f06f9e4` |
| [原冻结checker源码](../scripts/check_tracking_v5_state_updated_trajectory_control.py) | `5cd194a50740e1a35faebc655624f7db982f76a152ceee48e769626cb2c68e1c` |

按[新v2执行前登记](../plans/v0.0.1/state_updated_trajectory_check_v2_20261028.md)，新增独立源码 `scripts/check_tracking_v5_state_updated_trajectory_control_v2.py`，输出使用全新 `outputs/tracking_v5_state_updated_trajectory_control_v2_check_20261028`；新owner上限5400秒，以完整范围的实测成本留出检查时间，不改变任何控制/任务门槛。v2运行前要求原attempt的明确停止决策和同一child实际wait/reap证据，原producer仍必须完整通过；这不是无记录重启。

v2仅把每次query的有限同shape逐位检查改为 `np.array_equal`，把已检查有限/shape并计算最大绝对差的compare改为 `max_abs_difference <= tolerance`，替代成功路径的通用 `np.testing` 开销。全部scalar qpos/qvel和每个quaternion的原归一化条件仍在每次查询检查；原1e-10状态/几何/指标和1e-8力矩容差、所有动态方程、跨period递推、全trace分类/排序/预算保持。AST对照确认整个数值try主体除query检查外相同，包括重建、完整候选比较、约束和选择逻辑；compare仍在判定前累计最大差。执行前比较逻辑验证已完成：`tests/test_trajectory_audit_comparisons.py` 13项CPU测试通过（1.64s；临时目录`outputs/pytest_trajectory_audit_comparisons_20261028_001`），覆盖1e-10/1e-8正负ULP边界、不同量级、空数组/shape、NaN/inf/差值溢出、精确状态与四元数归一化；matching NaN/inf从旧exact接受变为新检查更早拒绝，未放宽有限数值接受范围。随后正式v2已完成全部范围，未借用原500条部分进度拼接结果。

v2源码/owner/比较测试提交为`75d2719`。本次独立重算48000个预测控制周期及1200000内步，每个period25次内环查询加1次端点查询，另加起点查询，共1248001次；全部4000条逐条核对TCP误差、约束分类、预算和排序，三份完整initial/selected/closest轨迹另核对全部q/v/goal/pose/twist/力矩/clipping。全trace TCP误差最大差2.86576318e-15、nonlinear safety_min差5.16955389e-15；三份完整候选q/v最大差5.55111512e-16/3.71924713e-15、requested/applied torque最大差9.20152843e-13。原状态/几何/指标1e-10与力矩1e-8容差保持。三份完整候选的q/v差值不能冒充另外3997份未归档完整状态数组的逐项比较。

独立原生JIT与旧纯Python AST的方向增量矩阵在36123个预测period有差异，最大1.19209290e-7；这是两种原生实现路径的单列诊断，不能拿纯Python近似替换正式JIT。producer与独立JIT重算之间的完整goal最大差7.77156117e-16，实际367历史latch核对差0。1030条rate违规差分/搜索探针全部计入响应预算、保留失败判定；没有据结果删候选或放宽任何门槛。

| 完整v2独立证据 | SHA256 |
| --- | --- |
| [checker verified](../outputs/tracking_v5_state_updated_trajectory_control_v2_check_20261028/verified.json) | `06750c34880e06b87332cbd79e9608f2bc7142a31cdef37ce2d8c2961d5a0f08` |
| [checker registration](../outputs/tracking_v5_state_updated_trajectory_control_v2_check_20261028/registration.json) | `42017e408e05a04c8a486e49b7d65c8f59b026c68fcfa2451367c63930c8dcde` |
| [owner result](../outputs/tracking_v5_state_updated_trajectory_control_v2_check_execution_20261028/result.json) | `927ea1c224dafd6a1ae64416f00a72600ef61b5c641c189bd8dbc317185d2d4e` |
| [owner registration](../outputs/tracking_v5_state_updated_trajectory_control_v2_check_execution_20261028/registration.json) | `f0663f7ac197fc6db881d647b7716eba4fe5157d7b17f73f04595f1bac395406` |

owner launch SHA `238d7abef6369ad711b4ee06b4576a4f71a454bea0261b77c4cca13438d669db`，child PID3061337/start_ticks612914299/parent3061134；CUDA/LD均为空、无自动重试。root持有的会话6677已实际取得owner exit0；child actual_wait=true/returncode0，release reaped=true/returncode0/signal_sent=false/original_error=null，身份连续。1761项owner来源、原producer1754项来源及checker1820引用均重新核对SHA一致，去重共1828路径。原v1的外部SIGTERM与失败退出记录保留，不能用v2成功掩盖。

下一机制已按[保留达标前缀修复计划](../plans/v0.0.1/prefix_preserving_trajectory_control_20261028.md)实现，solver/17项CPU组件测试提交`58440fa`，实验入口/独立checker/16项ledger测试提交`263ce1d`。新solver保留已达标前缀的原caps，所有72命令仍自由，每阶段仅修复首失败点；完整H12安全条件、全12点最终精度和共享4000/60预算不变。17项开发环境测试通过2.21s，16项ledger测试通过1.44s；两组又在正式Python3.11.15/NumPy1.26.4/SciPy1.15.3环境合计33 passed/9.90s，来源及实际exit0见[持久QA](../outputs/prefix_native_qa_20261028_vm4v8_yg/result.json)，SHA `738878bd5b9ed52a524ccafe19cf6b4e3bd8278412bac258a1b14e2e089a37f2`。这些合成组件和账本检查不构成本case的数值或实际控制资格。

v2完整闭环后，唯一新prefix producer已在CPU实际启动，root会话90660，child PID3117933/start_ticks613192333/parent3117803；输出`outputs/tracking_v5_prefix_preserving_trajectory_control_20261028`。当前仍在计算，无完整新搜索结果、独立数值资格或新物理执行。初始化、原生预测闭包和原数据保持；本轮只检验修复顺序是否能保住原连续6点并完成后续全部点。

本轮新增物理周期、世界模型调用和fresh case均为0。v2完整核验关闭但无witness，因此本轮不进入原生实际推进；新修复方案若有完整候选仍须独立复算，再另行登记实际前向。原18条正确参考、完整12条WAM开发、最终24条新初态及总任务目标保持，GPU继续禁止。
