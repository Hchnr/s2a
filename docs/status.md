# 当前目标与工作状态

本页维护当前事实，按日期的完整研究记录由索引定位。目录整理不改变研究目标。

用户后续已明确恢复 GPU 使用，并允许当前容器可见的四张卡；模型进程按实际 UUID
绑定，CPU/OSMesa 仍使用空 CUDA/LD。根 README 为迁移时固定字节，其旧禁用说明
应按当时的限制理解，最新用户许可覆盖该历史描述。

用户要求合适的阶段通过 PR 以 squash 方式合入 `main`，合并提交标题保留
`(#PR号)` 并核对远端关联；不直接向 `main` 推送提交。根 `AGENTS.md` 为迁移
清单固定字节，此约定记录于当前状态文档。

2026-11-13 三任务双 seed 自适应矩阵的六条实际轨迹均已独立重放：0/6 任务
成功，一条在 462 步耗尽请求预算且记录 346 个 grasp，另有 16 次 pose 超限的
案例；原审核失败保留。固定 H40 六案例短窗同为 0/6 抓取和成功，有 5 个 pose
超限端点。结果见[自适应矩阵](../reports/tracking/references/reset_multitask_adaptive_complete_20261113.md)
及[固定 H40 对照](../reports/tracking/references/reset_multitask_fixed_h40_complete_20261113.md)；
修复后的正式目录 64 并发相关回归 250 项通过，首次失败及复验已记录。

2026-11-13 统一固定 H40 对照的首四案例已完成 producer 及独立重放；三条走完
48 周期，一条在 36 周期质量停止，四条均未抓取或成功，停止案例有两个端点超
pose 限值。六案例尚未全部完成，见[首批结果](../reports/tracking/references/reset_multitask_fixed_h40_first_wave_20261113.md)。

2026-11-13 六案例首批四条在质量停止后因结束帧审核错误导致原 owner 失败；
修正后已从 reset 独立重放全部 359 周期，数值差为 0，四条均无目标 grasp 或
任务成功，wine/seed42 有 1 周期超 pose 限值。剩余两条仍在运行；正式目录完整
回归待做，见[事件与重放报告](../reports/tracking/references/reset_multitask_postflight_replay_20261113.md)。

2026-11-13 三任务 × 两 seed 的 600 周期统一开发入口已完成案例准入、模型和
仿真 worker、独立重放、四卡分批监督及 32 并发相关回归；最终准入和实际执行
尚未进行，见[准备报告](../reports/tracking/references/reset_multitask_adaptive_preparation_20261113.md)。

2026-11-13 已固定三任务 × 两 seed、每条最多 600 周期的统一开发协议；复位到
自适应流和独立重放的 32 并发相关回归 37 项通过。多案例现场执行尚未开始，见
[开发计划](../plans/v0.0.1/tracking/references/reset_multitask_adaptive_600_20261113.md)。

2026-11-13 seed 43 旧开发初态从 reset 实际执行 16 周期，其中后 8 周期使用本次
第 8 步图像生成的真实 WAM 参考；独立重放的动作、状态及内步数值差为 0。位姿
限值 16/16，目标 grasp 与任务成功均为 0；这不替代多任务和最终验收。后续优先
推进统一候选的多任务长时程对照，见[实际结果](../reports/tracking/references/reset_seed43_live_second_window_20261113.md)。

以下条目保留各阶段当时的状态，较早记录中的“尚未执行”以较新的结果为准。

2026-11-13 seed 43 两窗口的 producer、模型与独立 checker 已接入进程监督，
32 并发相关回归通过；最终准入和现场执行待完成，见[接线报告](../reports/tracking/references/reset_seed43_execution_20261113.md)。

2026-11-13 独立 checker 的 producer 来源、进程、GPU 与成本前置核验已通过
CPU 回归；父进程监督待完成，见[监督报告](../reports/tracking/references/reset_seed43_supervision_20261113.md)。

2026-11-13 当前图像模型交换审计已用历史真实响应及篡改测试验证；新 seed 43
现场运行待完成，见[模型交换审计](../reports/tracking/references/reset_seed43_model_exchange_audit_20261113.md)。

2026-11-13 seed 43 双窗口已接入有界中断成本记账并通过 CPU 回归；进程监督
和现场执行待完成，见[成本报告](../reports/tracking/references/reset_seed43_costs_20261113.md)。

2026-11-13 双窗口物理与独立重放审计已用历史收据及篡改测试验证；seed 43
新闭环尚未执行，见[审计报告](../reports/tracking/references/reset_seed43_feedback_audit_20261113.md)。

2026-11-13 seed 43 复位闭环接线的 64 并发相关回归通过；真实执行前仍需
进程监督及预检，见[阶段检查](../reports/tracking/references/reset_seed43_wiring_checks_20261113.md)。

2026-11-13 seed 43 producer 与独立 checker 的仿真接线已通过 CPU 合成回归；
现场进程监督与实际执行待完成，见[仿真接线报告](../reports/tracking/references/reset_seed43_simulation_worker_20261113.md)。

2026-11-13 seed 43 预检接线已绑定仿真实际图像目录并通过 CPU 回归；现场
进程监督及真实复位闭环待完成，见[预检接线报告](../reports/tracking/references/reset_seed43_trial_20261113.md)。

2026-11-13 seed 43 当前图像模型 worker 已通过 socket 合成测试与 CPU 回归；
实际模型与仿真仍待接入，见[worker 报告](../reports/tracking/references/reset_seed43_model_worker_20261113.md)。

2026-11-13 seed 43 当前图像单请求服务已完成 CPU 合成测试；模型进程接线、
实际推理和仿真待完成，见[服务报告](../reports/tracking/references/reset_seed43_live_request_20261113.md)。

2026-11-13 seed 43 复位至当前图像第二窗口的历史输入准入已通过 CPU 回归；
现场接线与实际试验待完成，见[准入报告](../reports/tracking/references/reset_seed43_live_admission_20261113.md)。

2026-11-13 新 seed 43 输出与历史同图参考逐元素一致；对应历史首 8 周期
位姿 8/8 达门但任务成功 0/8，该关联不是新在线闭环，见[只读审计](../reports/tracking/references/seed43_historical_control_20261113.md)。

2026-11-13 双 seed worker 已在旧开发图像上完成一次真实 seed 43 模型请求：
35 次去噪、进程退出和 GPU 释放均核对通过；没有物理闭环，见[重放报告](../reports/tracking/references/adaptive_multiseed_real43_20261113.md)。

2026-11-13 真实 seed 43 离线重放的进程监督已通过 CPU 合成回归；实际模型
推理尚未执行，见[监督准备](../reports/tracking/references/adaptive_multiseed_real43_owner_20261113.md)。

2026-11-13 真实 seed 43 的旧开发图像、记录端点和 checkpoint 清单已完成只读
准入；实际模型推理尚未执行，见[准入报告](../reports/tracking/references/adaptive_multiseed_real43_admission_20261113.md)。

2026-11-13 双 seed 自适应请求循环与模型 worker 已完成 CPU 合成接线和回归；
真实模型采样和物理执行尚未进行，见[worker 报告](../reports/tracking/references/adaptive_multiseed_worker_20261113.md)。

2026-11-13 双 seed 模型请求与响应身份合同已通过 32 并发回归；服务循环和
真实采样尚未接入，见[合同报告](../reports/tracking/references/adaptive_multiseed_contract_20261113.md)。

2026-11-13 已登记双 seed 自适应模型接口，解决现有服务固定 seed 42 的验收
缺口；实现与真实 seed 43 验证尚未完成，见[计划](../plans/v0.0.1/tracking/references/adaptive_multiseed_interface_20261113.md)。

2026-11-13 保存状态静态重建与 40 个原生标签一致；第 129 端点位置条件成立
但无碗盘接触。原盘位置向量记录缺陷已只读纠正，见[诊断报告](../reports/tracking/references/bowl_native_predicate_probe_20261113.md)。

2026-11-13 碗任务诊断 QA 中两次合成 MuJoCo 测试各创建一个小模型并 forward；
此前零模型表述已更正，真实状态查询未执行，见[计划修订](../plans/v0.0.1/tracking/references/bowl_native_predicate_probe_20261113.md)。

2026-11-13 碗任务谓词的静态诊断入口已通过 32 并发回归；实际查询尚未执行，
见[准备报告](../reports/tracking/references/bowl_native_predicate_probe_preparation_20261113.md)。

2026-11-13 已登记基于 40 个保存状态的原生任务谓词静态诊断；尚未执行，见
[诊断计划](../plans/v0.0.1/tracking/references/bowl_native_predicate_probe_20261113.md)。

2026-11-13 最终端点缓存 checker 完整重放 147 周期且数值差为 0；第 129 端点
任务失败、末尾连续 18 周期成功，见[独立重放报告](../reports/tracking/references/adaptive_long_final_endpoint_replay_20261113.md)。

2026-11-13 最终端点 checker 的首次 64 并发回归中 2 项因获取当前目录失败；
单项复验通过但原因未确定，完整回归待独立重验，见[计划修订](../plans/v0.0.1/tracking/references/adaptive_long_final_endpoint_checker_20261113.md)。

2026-11-13 最终端点 checker 的监督和失败记账已通过 32 并发回归；真实重放
尚未执行，见[监督报告](../reports/tracking/references/adaptive_long_final_endpoint_owner_20261113.md)。

2026-11-13 最终端点 checker 已通过接线回归，较早观察保持原路径；真实重放
尚未执行，见[接线报告](../reports/tracking/references/adaptive_long_final_endpoint_worker_20261113.md)。

2026-11-13 最终端点缓存重放的历史准入已通过 32 并发回归，真实重放尚未
执行，见[准入报告](../reports/tracking/references/adaptive_long_final_endpoint_admission_20261113.md)。

2026-11-13 已登记仅在第 147 端点启用严格缓存的独立 CPU 重放；尚未实现或
执行，见[计划](../plans/v0.0.1/tracking/references/adaptive_long_final_endpoint_checker_20261113.md)。

2026-11-13 长窗缓存 checker 完成 119 周期后在状态哈希核对处失败；第三组
图像字节一致，原失败和本次成本均保留，见[事件报告](../reports/tracking/references/adaptive_long_cached_checker_incident_20261113.md)。

2026-11-13 长窗缓存 checker 的单次 CPU 监督和失败成本记录已通过 32 并发
相关回归；真实重放尚未执行，见[监督报告](../reports/tracking/references/adaptive_long_cached_checker_owner_20261113.md)。

2026-11-13 长窗缓存 checker 的独立重放接线已通过 32 并发相关回归；真实
重放尚未执行，见[接线报告](../reports/tracking/references/adaptive_long_cached_checker_worker_20261112.md)。

2026-11-12 长窗缓存 checker 已通过真实历史及部分审计准入的 32 并发回归；
缓存接线和独立重放待完成，见[准入报告](../reports/tracking/references/adaptive_long_cached_checker_admission_20261112.md)。

2026-11-12 已登记长窗失败后的独立缓存 checker 验证；实现和真实重放尚未执行，
见[计划](../plans/v0.0.1/tracking/references/adaptive_long_cached_checker_20261112.md)。

2026-11-12 40 周期长窗生产至第 147 周期；第 129 端点任务成功为假，末尾连续
18 周期成功。checker 完成动作后额外端点读取失败，已保存部分独立审计，见
[实际报告](../reports/tracking/references/adaptive_live_bowl_long_partial_20261112.md)。

2026-11-12 40 周期长窗运行入口和跨解释器登记已通过 32 并发回归；最终回归、
预检与真实执行待完成，见[阶段报告](../reports/tracking/references/adaptive_live_bowl_long_owner_20261112.md)。

2026-11-12 可配置生产者/checker 监督及中断成本保留已通过 32 并发回归；
长窗试验接线和真实执行待完成，见[阶段报告](../reports/tracking/references/registered_adaptive_execution_20261112.md)。

2026-11-12 可配置预检和 40 周期因果账本边界已通过 32 并发回归；进程监督及
真实长窗仍待完成，见[阶段报告](../reports/tracking/references/registered_adaptive_preflight_20261112.md)。

2026-11-12 长窗试验准入已复核前次独立重放和累计区间，通过 32 并发回归；
监督及真实执行待完成，见[准入报告](../reports/tracking/references/adaptive_live_bowl_long_admission_20261112.md)。

2026-11-12 已预登记同一开发初态的 40 周期连续重规划试验；新登记、监督和
真实执行仍待完成，见[计划](../plans/v0.0.1/tracking/references/adaptive_live_bowl_long_20261112.md)。

2026-11-12 独立恢复试验在一个旧开发初态完成 115 周期，末尾 14 周期原生任务
成功；独立重放逐值一致。末尾 40 周期、24 条留出及实时资格仍未达到，见
[实际结果](../reports/tracking/references/adaptive_live_bowl_recovery_20261112.md)。

2026-11-12 独立恢复试验监督已通过跨解释器检查和 32 并发回归；最终回归、
预检及真实试验仍待完成，见[阶段报告](../reports/tracking/references/adaptive_live_bowl_recovery_owner_20261112.md)。

2026-11-12 恢复试验的独立物理审核及累计区间已通过历史收据和篡改检查；
新进程监督与实际试验待完成，见[审核报告](../reports/tracking/references/adaptive_live_bowl_recovery_audit_20261112.md)。

2026-11-12 独立恢复试验的配置、旧失败范围和来源准入已通过 32 并发回归；
进程监督及真实运行仍待完成，见[准入报告](../reports/tracking/references/adaptive_live_bowl_recovery_admission_20261112.md)。

2026-11-12 已为来源修复后的单开发案例登记独立试验及累计资源区间；执行器、
最终回归和真实试验仍待完成，见[计划](../plans/v0.0.1/tracking/references/adaptive_live_bowl_recovery_20261112.md)。

2026-11-12 已将无绑定历史输入与旧失败收据、未知开销区间合并准入；新试验
配置和监督仍待接线，见[报告](../reports/tracking/references/adaptive_recovery_inputs_20261112.md)。

2026-11-12 已实现无全局绑定的 step107 模型输入加载，并与历史加载结果独立
对照；新试验监督器尚未接入，见[报告](../reports/tracking/references/step107_model_inputs_20261112.md)。

2026-11-12 可中断自适应仿真 worker 已完成合成接线与 32 并发回归；新试验的
配置和进程监督仍待接入，见[报告](../reports/tracking/references/managed_adaptive_simulation_20261112.md)。

2026-11-12 已将可处理中断和部分物理计数接入独立 worker 封装，并通过 32 并发
回归；新试验运行器尚未接线，见[阶段报告](../reports/tracking/references/managed_physical_worker_20261112.md)。

2026-11-12 中断处理与物理开销区间组件已通过真实信号和 32 并发回归；新仿真
worker 尚未接入，见[阶段报告](../reports/tracking/references/interrupted_physical_costs_20261112.md)。

2026-11-12 已以实际模型 Python 无 GPU 地复核旧失败注册和完整来源；独立准入
组件及专项回归通过，新执行器尚未接入，见[报告](../reports/tracking/references/registered_model_bundle_20261112.md)。

2026-11-12 已登记来源故障后的恢复设计，并为缺失的物理收据保留累计开销区间；
新执行器和新的独立试验仍待准备，见[设计](../plans/v0.0.1/tracking/references/adaptive_live_bowl_recovery_design_20261112.md)。

2026-11-12 已实现绑定顺序无关的模型来源清单并通过跨解释器和 32 并发检查；
新执行器尚未接入，见[阶段报告](../reports/tracking/references/adaptive_live_bowl_source_manifest_20261112.md)。

2026-11-12 连续重规划生产阶段因新进程来源清单不一致而在模型加载前失败；
模型请求为零，仿真 worker 无终止收据，其物理消耗保留为未知，见[失败记录](../reports/tracking/references/adaptive_live_bowl_admission_failure_20261112.md)。

2026-11-12 连续重规划监督入口已完成 CPU 专项回归，并保留失败时的未知模型
开销；最终并发回归、预检和真实试验仍待完成，见[阶段报告](../reports/tracking/references/adaptive_live_bowl_supervisor_20261112.md)。

2026-11-12 连续重规划的实际收据审核已用历史独立物理记录和篡改测试验证；新
试验仍未启动，见[审核报告](../reports/tracking/references/adaptive_live_bowl_audit_20261112.md)。

2026-11-12 连续重规划的独立仿真 worker 已通过 CPU 准入和来源检查；外层监督器
与真实试验仍待完成，见[阶段报告](../reports/tracking/references/adaptive_live_bowl_worker_20261112.md)。

2026-11-12 连续重规划已核对历史输入和累计开销，修正仿真与模型并发启动时的
目录时序；真实试验仍未启动，见[准入报告](../reports/tracking/references/adaptive_live_bowl_admission_20261112.md)。

2026-11-12 连续重规划真实模型 worker 的设备与预算准入已通过 CPU 检查；进程
所有者、预检及真实试验仍待完成，见[阶段报告](../reports/tracking/references/adaptive_model_worker_20261112.md)。

2026-11-12 已以合成服务接通 reset 到连续重规划及独立重放；真实进程所有者和
预检尚未完成，见[阶段报告](../reports/tracking/references/adaptive_live_bowl_bridge_20261112.md)。

2026-11-12 已登记最多两次模型请求的连续重规划开发试验，并实现任意已完成周期
的内步边界；真实试验尚未执行，见[计划](../plans/v0.0.1/tracking/references/adaptive_live_bowl_20261112.md)。

2026-11-12 连续重规划的独立变长 checker 已通过合成重放和来源边界检查；实际模型
及仿真进程所有者尚未接入，见[阶段报告](../reports/tracking/references/bounded_replan_replay_20261112.md)。

2026-11-12 修正连续重规划在预算结束与关节缓冲不足同时出现时的停止原因优先级，
见[阶段报告](../reports/tracking/references/bounded_replan_safety_priority_20261112.md)。

2026-11-12 连续重规划的模型请求循环与控制通道已用合成推理联调；实际模型
进程所有者和独立 checker 尚未接入，见[阶段报告](../reports/tracking/references/bounded_replan_model_loop_20261112.md)。

2026-11-12 连续重规划的当前图像与部分段反馈绑定通过合成回归；真实长期模型
服务和独立 checker 尚未接入，见[阶段报告](../reports/tracking/references/bounded_replan_reference_20261112.md)。

2026-11-12 连续因果重规划的变长控制执行器通过合成时序及故障检查；真实模型、
实际分段策略和独立重放尚未接入，见[阶段报告](../reports/tracking/references/bounded_replan_runtime_20261112.md)。

2026-11-12 连续因果重规划的变长段协议及 32 并发专项检查已完成；尚未接入实际
运行器或真实模型，见[阶段报告](../reports/tracking/references/bounded_replan_protocol_20261112.md)。

2026-11-12 一次并行专项检查在启动前因误改根 `AGENTS.md` 的固定字节而停止，
未启动测试 worker；失败收据保留在 `outputs/qa/replayed_reference_pose_bounds_20261112/`
（SHA-256 `c2a93340652f5edc3e9a6f0520b2bbec7ef8c10c530e39add93c9b3345a43090`）。
现已恢复清单登记的根文件字节，后续使用独占的新检查目录。

2026-11-12 因果提前重规划在同一旧开发初态执行至 step113，新参考完成 6/40
目标后质量停止；113 周期独立重放逐值一致，新窗口成功不连续，见[实际执行报告](../reports/tracking/references/causal_replan_step107_execution_20261112.md)。

2026-11-12 已对核验的 step107 当前图像完成一次真实 WAM 请求，响应与图像、
状态绑定且 GPU 已释放；新参考尚未执行。见[模型报告](../reports/tracking/references/causal_replan_step107_model_20261112.md)。

2026-11-12 已从 reset 独立重放 107 周期并采集因果重规划边界的当前图像；与
原轨迹的命令、状态和内步逐值一致，尚未生成或执行新参考。见[采图报告](../reports/tracking/references/causal_replan_step107_capture_20261112.md)。

2026-11-12 已实现保持完成历史的部分窗口因果换段；对碗任务同一旧开发轨迹
只读影子检查首次在 step107 触发，尚未接入当前图像或物理动作。见[阶段报告](../reports/tracking/references/causal_partial_replan_20261112.md)。

2026-11-12 对碗任务 step108 的已重放停止状态单独运行首点优先求解：320 次
模型响应、66 个线性安全尝试中没有首点双阈值候选；未派发动作或独立审核候选，
详见[对照报告](../reports/tracking/joint_control/replayed_first_pose_search_20261112.md)。

2026-11-12 对碗任务 step104、106、107、108 的已重放状态计算冻结模型首点
外界；所选投影均未排除原首点双阈值，不能据此判定可行或物理不可达。首轮软件
库登记失败及零模型开销保留，修复后的独立复核见[结果](../reports/tracking/joint_control/replayed_reference_pose_bounds_20261112.md)。

2026-11-12 碗任务同一开发 reset 依据实际当前图像 WAM 参考连续执行至 step108，
端点 102–108 的原生任务成功标签均为真；端点 108 位置误差 10.73 mm，随后下一
动作前因质量无 witness 停止。108 周期独立重放一致，仍未满足留出 600 周期及末尾
40 周期成功要求；见[物理报告](../reports/tracking/references/reset_bowl_step88_reference_20261112.md)。

2026-11-12 在独立重放相同 88 周期后已采集当前图像，并生成下一条真实 WAM 参考；
该参考尚未执行，见[模型报告](../reports/tracking/references/reset_bowl_step88_model_20261112.md)
和[采图报告](../reports/tracking/references/reset_bowl_step88_capture_20261112.md)。

2026-11-12 已用通用尾段组件连续执行并独立重放碗任务同一开发 reset 的 88 周期：
第二条 WAM 参考 40 个目标全部执行，端点 64–88 的原生目标 grasp 标签均为真，
任务成功仍为 0；见[完整参考报告](../reports/tracking/references/reset_bowl_h40_tail_20261112.md)。

2026-11-12 已新增显式预算上限为 600 周期的 reset 运行器，旧 48 周期实现保持
不变；同一开发 reset 已连续执行并独立重放 56 周期。第二 WAM 参考前 8 周期
位姿 8/8 达标、目标 grasp / 任务成功均 0，见[物理短窗报告](../reports/tracking/references/reset_bowl_h40_second_window_20261112.md)
与[长轨迹设计](../plans/v0.0.1/tracking/references/long_horizon_reset_20261112.md)。

2026-11-12 碗任务固定 H40 的 48 周期经独立重放后，已采集当前双相机图像并据此
生成真实 WAM H40 参考；新参考前 8 个开度目标从约 0.80 降到 0.40，尚未物理执行。
此前轨迹的 grasp / 任务成功仍为 0，见[模型报告](../reports/tracking/references/reset_bowl_h40_reobserve_model_20261112.md)
及[图像报告](../reports/tracking/references/reset_bowl_h40_reobserve_capture_20261112.md)。

2026-11-12 碗任务加强闭合探针在 45 周期后因质量无 witness 停止，目标 grasp /
任务成功均为 0；最后两端点位置超 1 cm，端点 45 一根手指略超模型标称范围。
独立重放一致，不提升为默认策略，见[闭合探针报告](../reports/tracking/gripper/bowl_h40_closing_intent_20261112.md)。

2026-11-12 双 pad 静态闭合恢复枚举在两条旧开发轨迹的末端各发现 8 个满足原
1 cm TCP 位置门的几何 witness，但对应 opening 约 0.047–0.151，与原模型 0.706
差异明显；未验证物理抓取。见[闭合恢复报告](../reports/tracking/gripper/bowl_static_close_feasibility_recovery_20261112.md)。

2026-11-12 固定姿态双 pad 闭合枚举在第 11 个 LP 的残差检查停止；单组合复现显示
默认求解容差与 `1e-8 m` 本地检查不匹配，显式 `1e-9` 后该组合通过残差检查。
首批枚举未完成；后续恢复结果见上文，[LP 诊断](../reports/tracking/gripper/bowl_static_close_lp_diagnostic_20261112.md)保留原因。

2026-11-12 对两条碗任务轨迹所选八个保存构型，离线测试 site 负 x 方向 2 cm
固定平移：无一形成双侧 pad 与碗 box 严格重叠；1,280 个 box 对经独立线性规划
复核。仅为单案例静态检查，见[偏移几何报告](../reports/tracking/gripper/bowl_static_site_offset_20261112.md)。

2026-11-12 修复并独立复核指垫快照：固定 H40 和开度提前分支第 48 周期末，
右 pad 与碗 box 沿夹爪轴分别分离约 1.62 cm / 1.80 cm；提前分支在周期 47、48
仅左 pad 接触。该离线结果不提供在线物体真值，见[几何恢复报告](../reports/tracking/gripper/bowl_pad_alignment_recovery_20261112.md)。

2026-11-12 指垫几何诊断的事后审核发现快照数组被后续 forward 覆盖，首批结果
不具备几何结论资格，失败记录和实际成本已保留；见[快照失败报告](../reports/tracking/gripper/bowl_pad_alignment_snapshot_failure_20261112.md)。

2026-11-12 碗任务固定 H40 将 opening 提前 4 周期的开发对照，两组各 48 周期
均无目标 grasp 或任务成功；候选相对原 WAM opening 的 RMSE 从 0.024659 增至
0.099420，不提升为默认策略。独立重放一致，见[开度对照](../reports/tracking/gripper/bowl_h40_opening_lead_20261112.md)。

2026-11-12 对两条碗任务轨迹的 96 个已保存端点重建接触：任一指垫与碗
均无接触对，固定 H40 后段出现碗与手指其他部位及手掌接触对；重建双侧接触
与原 grasp 标签一致。仅覆盖端点、不证明内步或接触力，见[接触诊断](../reports/tracking/references/reset_bowl_saved_contacts_20261112.md)。

2026-11-12 依据两条保存状态重建碗任务几何：第 48 周期末 TCP 到碗刚体原点的
距离，固定 H40 与重观测分支分别约 0.0726 m 和 0.2248 m，均无已记录 grasp。
原点距离不等于表面接触距离；前一几何诊断的状态逐位检查失败和两次开销均保留，
见[保存状态几何报告](../reports/tracking/references/reset_bowl_saved_geometry_recovery_20261112.md)。

2026-11-12 碗任务从同一 reset 和 step8 WAM 响应固定消费完整 H40：实际开度最低
约 0.7669，仍无目标 grasp/任务成功；与重观测分支的前 16 周期及独立重放均一致。
这是一个旧初态短窗的协议对照，见[固定 H40 报告](../reports/tracking/references/reset_bowl_fixed_h40_probe_20261112.md)。

2026-11-12 对三任务恢复试验的 12 条真实 WAM 响应做只读检查：90 个实际消费的
夹爪参考均没有明显闭合；5 个相对当前开度下降至少 0.2 的端点仅出现在碗任务的
未消费输出末尾。关联分析不证明唯一原因，见[夹爪意图诊断](../reports/tracking/references/reset_multitask_gripper_intent_audit_20261112.md)。

2026-11-12 三任务旧开发初态从 reset 使用实际当前图像重复请求 WAM：两个案例各
完成 48 周期，奶酪案例在 step18 因 `quality_no_witness` 停止；三个案例均无
目标 grasp/任务成功标签。114 周期及 12 张当前图像均通过独立重放与审计；模型
平均耗时约 12.20 s，物理等待时钟暂停，尚非实时。前一轮计数器错误及实际消耗
单独保留，见[恢复报告](../reports/tracking/references/reset_multitask_live_recovery_20261111.md)
和[失败审计](../reports/tracking/references/reset_multitask_live_failure_20261111.md)。

2026-11-11 从一个开发 reset 完成 8 周期后，用本次 step8 当前图像请求真实 WAM，
并连续执行新参考的 8 周期；新窗口 8/8 位姿达标、grasp/任务标签 0/8，修复后的
独立重放数值最大差为 0。仅一个旧初态，模型耗时约 13.24 s；原 checker 失败和
成本保留，见[当前图像报告](../reports/tracking/references/reset_live_second_window_20261111.md)。

2026-11-11 从复位建立可换段的手臂、夹爪及内环历史；在 12 条相关开发案例中，
首 8 周期的新策略、原运行和独立重放的命令/状态/内环最大数值差均为 0。尚未执行
第二个新图像参考或完整任务，见[首窗口等价报告](../reports/tracking/joint_control/segmented_reset_equivalence_20261111.md)。

2026-11-11 从实际复位启动的三任务 12 条相关开发案例各执行并独立重放 8 周期；
新方案和同参考旧方案均为 96/96 位姿端点达标，新方案所有内环余量达标，但两者
均无 grasp/任务成功标签，新方案误差略高，决策中位 103 ms 尚非 20 Hz 实时。
仅有三个旧开发初态，不构成整任务或留出验收；见[首窗口反馈报告](../reports/tracking/joint_control/reset_start_feedback_20261111.md)。

2026-11-11 三任务 12 条开发案例的现场复位和首图在修复插值线程差异后均与原
记录逐字节匹配；原 12 条图像哈希不符的失败和额外开销保留，独立产物审核通过。
本阶段没有派发动作或推进物理，详见[现场复位报告](../reports/tracking/joint_control/reset_native_admission_20261111.md)。

2026-11-11 从复位启动的开发对照已完成空历史实现和 12 条原首请求来源自检；
尚未重建现场、核验现场图像或执行任何候选动作，不具备控制资格。下一步按固定
预算接入现场复位与独立重放，详见[准备报告](../reports/tracking/joint_control/reset_start_preparation_20261111.md)。

2026-11-11 将 finger2–桌面充分碰撞检查扩展到三任务 v4 开发矩阵的 7,200 个
已生成目标：固定查询时关节位置有 9 个证书，整个额定关节行程为 0 个；未获证
不表示无碰撞。三个任务仅三个开发初态，不能视作留出成功率；9 个肯定证据
已独立复核。见[多任务条件检查](../reports/tracking/response_models/multicase_slide_table_collision_20261111.md)。

2026-11-11 对两类已审核 WAM 参考的 30 个相关端点进行条件几何证书检查：
11 个端点即使允许整个额定夹爪行程及原位姿容差，仍有 finger2 顶点必然位于
固定仿真桌面内部；其余端点未被证明，并非无碰撞结论。见[行程碰撞证书报告](../reports/tracking/response_models/slide_range_table_collision_20261111.md)。

2026-11-11 提前投影组的 finger1–碗接触后，六周期末碗世界平移约 2.645 mm、
旋转约 0.0341 rad，奶酪相对碗平移约 0.027 mm；两组瞬时任务标签仍为真。
这是单一仿真短窗相关性，不能单独归因或证明长期稳定，见[物体运动报告](../reports/tracking/response_models/early_box_face_bowl_effect_20261111.md)。

2026-11-11 同起点、同 step112 WAM 参考的提前干预对照中，共同六周期的
finger2–桌面接触从 22/150 降为 0/150 内步，但投影组出现 finger1–碗
45/150 接触，原参考位姿均为 2/6 达标且投影组平均误差更大。投影组随后因
修正超上限停止，独立重放一致；见[提前对照报告](../reports/tracking/response_models/early_box_face_feedback_20261111.md)。

2026-11-11 同起点、同 WAM 参考的五周期物理对照中，多方向投影把 finger2–桌面
有效接触从 32/125 降到 0/125 内步，但原参考位姿达标从 2/5 降至 1/5，所选
目标仅 0/5 达标；第六周期因修正略超 5 cm 上限在派发前停止。独立重放一致，
该短窗尚非任务验收，见[物理对照报告](../reports/tracking/response_models/box_face_feedback_20261111.md)。

2026-11-11 对相同已审核分支做多方向静态分离面分析：14 个关节范围内端点
有 12 个候选落在 5 cm 修正上限内，单纯向上为 4 个；全部修正仍超过原参考
1 cm 容差。两处关节越界继续拒绝计算，尚未执行新策略；见[多方向投影报告](../reports/tracking/response_models/box_face_projection_20261111.md)。

2026-11-11 对相同 16 个历史端点利用 finger2 的单关节滑动方向收紧条件包络：
14 个范围内端点中 4 个的联合投影可落在 5 cm 上限内，原各向同性界为 0 个；
另两个关节已越界的端点仍拒绝计算。全部可计算端点的竖直修正仍超过原参考
1 cm 容差，尚无新物理执行；见[方向包络报告](../reports/tracking/response_models/directional_slide_boxes_20261111.md)。

2026-11-11 对两条已审核分支的 16 个历史端点进行静态条件分析：联合考虑桌面与
炉具时，零夹爪形变条件下仅 4/16 个候选可在 5 cm 修正上限内投影；采用夹爪
全关节范围条件时 14 个可计算端点均超上限，另两个端点当前关节位置已略超额定
范围。分析未执行新投影轨迹，详见[联合几何报告](../reports/tracking/response_models/parallel_boxes_shape_bound_20261111.md)。

2026-11-11 从 step118 提前分叉后，使用两条分支各自的 step120 当前图像分别
生成真实 WAM 参考，并各执行八个物理反馈周期及独立重放：原目标与位姿包络
分支的 finger2–桌面接触分别为 107/200 和 0/200 内步，但投影分支仍有 83/200
内步接触炉具，原 WAM 位姿双容差分别仅 2/8 和 1/8 达标。两分支后续 WAM 参考
不同，不能把差异单独归因于投影。首次继续执行的 6,000 失败内步已保留并计入
累计 30,800 内步；模型请求耗时约 13.24 / 11.71 秒，未达实时。
详见[当前图像反馈报告](../reports/tracking/references/early_tabletop_live_feedback_20261111.md)。

2026-11-11 从 correct_08 的同一 step120 状态出发，在统一单步 OSC 反馈下实际
执行七周期并独立重放：原目标、零误差和位姿包络投影分别有 156/175、156/175、
81/175 个 finger2–桌面有效接触内步；包络分支仍有 123/175 个内步接触炉具燃烧区，
且 0/7 端点达到自身投影目标的原位姿双容差。三组原目标均为 2/7，夹爪实际
形变也不满足静态零形变假设。该短窗口尚不构成任务验收，详见
[物理对照报告](../reports/tracking/response_models/tabletop_projection_feedback_probe_20261111.md)。

2026-11-11 在 47 个已记录状态上，两个可选桌面参考投影的静态目标均消除了
原目标 step120–126 的 finger2–桌面接触，代价是最大抬升分别约 3.0 cm、
4.3 cm；后续实际反馈并未验证其为可靠避碰方案，见
[静态投影报告](../reports/tracking/response_models/tabletop_reference_projection_20261111.md)。

2026-11-11 在固定当前夹爪关节配置和桌面盒体的条件下，step123–126 已生成
目标存在保守碰撞证书：即使 TCP 位姿在原 0.01 m / 0.05 rad 容差内变化，
夹爪碰撞 mesh 的一个点仍在桌体内部。step120–122 未判定；改变夹爪开度或
场景后的可行性未知。详见[容差证书报告](../reports/tracking/response_models/reference_table_tolerance_20261111.md)。

2026-11-11 固定当前夹爪配置，将 correct_08 的已生成目标对齐到静态几何代理：
step120–126 的目标代理与桌面相交，MuJoCo 接触距离从 −0.000366 m 变化至
−0.029104 m；step126 原本未派发。该检查不证明 IK、连续路径或容差内不可行，
见[参考几何报告](../reports/tracking/response_models/reference_table_geometry_20261111.md)。

2026-11-11 对两条已审核物理轨迹做内步力矩与接触仪表化重放：step120 夹爪
开始接触桌面，step122 两分支均出现关节 6 力矩裁剪；随后历史分支接触力与
模型误差继续增大。两分支的 1,050 个共同前缀内步及各自完整重放一致。这是
仿真中的相关证据，尚未隔离唯一原因或验证避让策略，见[诊断报告](../reports/tracking/response_models/replayed_contact_torque_20261111.md)。

2026-11-11 对三任务已有物理记录做固定范围回顾：9 条相关轨迹共 103 周期，
其中 94 周期有本轨迹历史。上一周期 TCP 偏差补偿的假达门由 6 次降至 1 次，
同时出现 1 次假停止；余留假达门说明不能作为实际质量保证。各分支共享初态或
参考，且动作经过质量筛选，不能视作独立泛化评测。见[跨轨迹分析](../reports/tracking/response_models/cross_trajectory_tcp_innovation_20261111.md)。

2026-11-11 首次物理位姿失准的 step122，使用本次 step121 已完成偏差重新求解
并实际执行一周期；位置误差从历史同起点 0.013933 m 降至 0.012952 m，仍超
0.01 m 门，旋转误差为 0.025869 rad。独立重放一致，step123 为预登记预算停止，
不是证明无解。没有新 WAM 推理，详见[物理对照](../reports/tracking/joint_control/early_causal_innovation_physical_20261111.md)
与先前[静态对照](../reports/tracking/joint_control/early_causal_innovation_quality_20261111.md)。

2026-11-11 在 step125 冻结状态加入 step124 已完成 TCP 偏差后的质量约束，
69 次模型响应未找到同时满足原门和补偿门的候选；线搜索失败不证明全局不可行。
未派发动作，见[因果补偿质量搜索](../reports/tracking/joint_control/causal_innovation_quality_search_20261111.md)。

2026-11-11 单条已重放物理轨迹的 45 个有历史周期中，原模型预测 4 次假达门，
上一周期 SO(3)/位置偏差补偿 0 次；历史最大偏差界并未覆盖所有未来误差。该
离线比较不代表改变策略后的闭环效果，见[因果偏差分析](../reports/tracking/response_models/causal_tcp_prediction_audit_20261111.md)。

2026-11-11 六份已保存 WAM 响应的物理反馈中，数值修正使原 step125 停止后多
执行 1 周期，并在 step126 再次受控停止；末尾任务连续成功 39 个端点。新增
周期实际误差 0.024850 m / 0.081177 rad，未达 0.01 m / 0.05 rad 门；独立重放
全部 46 周期一致。没有新 WAM 推理，见[物理反馈报告](../reports/tracking/joint_control/quality_rate_repair_recorded_feedback_20261111.md)。

2026-11-11 可注入质量求解器在同一 stop125 状态自动找到并独立核验修正后的
局部模型候选，共 59 次模型响应；预测位置仅在原容差内达门，默认在线路径和
物理执行尚未改变。详见[自动求解报告](../reports/tracking/joint_control/quality_rate_repair_solver_20261111.md)。

2026-11-11 对已登记静态候选做 ≤`4.96e-10` 的动作数值修正后，独立冻结模型
核验通过原质量及关节/制动门；预测位置仅在原数值容差内达到 0.01 m，无实质
裕量。尚未派发或验证实际反馈，见[修正报告](../reports/tracking/joint_control/cached_live_stop125_rate_repair_20261111.md)。

2026-11-11 静态质量分支缓存复核定位到数值边界：58 个候选中 4 个仅因动作
变化率后置检查被拒，最终候选超过名义 0.25 约 `6.85e-12`；未放宽阈值或派发。
这只解释复核分支，非原真实 WAM 停止的精确重现；见[约束审核](../reports/tracking/joint_control/cached_live_stop125_constraint_audit_20261111.md)。

2026-11-11 step125 静态诊断中，关节约束候选通过独立模型核验，但下一周期位置
误差 0.014189 m 超过 0.01 m 质量门；三次质量约束搜索均无合格 witness，尽管
优化器报告成功终止，当轮未记录具体被拒约束。没有新增物理或 WAM 调用；详见
[静态诊断](../reports/tracking/joint_control/cached_live_stop125_feasibility_20261111.md)。

2026-11-11 端点缓存接入真实 WAM 后完成 6 次请求、210 次去噪与 45 个新反馈周期；
step125 因无合格质量 witness 在派发前受控停止。独立 checker 重放全部 45 条动作，
末尾连续 38 个任务成功端点，距原 40 端点门仍差 2 个；最后窗口位姿门 2/5，
模型单次约 11.75–13.31 s，尚不具备实时资格。详见[结果报告](../reports/tracking/references/causal_stream_live_cached_20261111.md)。

2026-11-11 已在 CPU/OSMesa 上用端点缓存重放同一 40 条动作并捕获 step120 当前
图像。原 owner 因误要求跨运行状态哈希一致而失败；只读审计及修正后的汇总核对
通过，完整状态最大跨运行差 `6.61e-13`，采集前后当前状态哈希一致。没有新的
WAM 推理或第六窗口，详见[缓存回放报告](../reports/tracking/references/causal_stream_endpoint_cache_replay_20261111.md)。

2026-11-11 最多六窗口的真实 WAM 开发试验在第五个窗口后异常中止：实际完成
5 次请求、175 次去噪和 40 周期反馈；下一端点观测触发物理状态一致性检查，
未执行第六次请求。后续 CPU 动作重放复现异常，40 条动作及全部内环样本与原记录
一致，末尾连续 33 个任务成功端点；不构成 40 端点稳定资格。GPU 已释放，见
[失败报告](../reports/tracking/references/causal_stream_live_48_partial_20261111.md)和
[诊断报告](../reports/tracking/references/causal_stream_endpoint_diagnostic_20261111.md)。

2026-11-11 已在同一次运行中使用真实 WAM 的两次当前图像推理完成 16 周期
CPU/OSMesa 反馈及独立 checker；末尾任务连续成功 9 周期，模型处理约 13.284/11.713 s，
仿真等待时暂停。仍未达到实时或 40 周期稳定门，见[真实模型报告](../reports/tracking/references/causal_stream_live_wam_20261111.md)。

2026-11-11 新流式执行器使用两份已保存 WAM 响应完成 CPU/OSMesa 16 周期物理
复现及独立 checker；末尾任务谓词连续 9 周期，未达 40 周期稳定门，且没有新模型
推理。详见[物理复现报告](../reports/tracking/joint_control/causal_stream_recorded_physical_20261111.md)。

2026-11-11 连续执行适配和独立命令重放已在 CPU 假环境验证，64 路相关回归
114 项通过；真实 WAM、OSMesa 连续执行及留出验收未完成，见[执行报告](../reports/tracking/joint_control/causal_stream_execution_20261111.md)。

2026-11-11 连续 WAM 流完成 CPU 协议与双子进程模拟：64 路相关专项 89 项
通过；真实模型、物理 runtime 和独立 checker 尚未接入，见[协议报告](../reports/tracking/joint_control/causal_stream_protocol_20261111.md)。

2026-11-11 用户已允许并鼓励恢复GPU，并确认当前容器可见的四张卡均可用；
当前UUID和首个推理检验登记见[计划](../plans/v0.0.1/tracking/references/gpu_resume_repro_20261111.md)。
H100 上一次 35 步 WAM 推理的两套 action 数组与历史开发样本逐元素一致，进程
退出后显存已释放；详情见[报告](../reports/tracking/references/gpu_resume_repro_20261111.md)。
已停止的旧任务和部分产物保留，不自动重跑。
记录通知时曾改动迁移清单固定的根文件，已恢复原字节；首次影子CPU准入在测试
收集前失败并保留收据，详见[修复记录](../plans/v0.0.1/tracking/gripper/defer_close_normal_shadow_qa_repair_20261111.md)。

原夹爪控制器18条正确参考的4907周期影子只在correct_08 step85触发；该case
两版原基线的动作、状态和夹爪决策一致，实际候选完整反馈仍未验证。机械臂合同
与task1质量反馈不同，见[影子报告](../reports/tracking/gripper/defer_close_normal_correct_shadow_20261111.md)。
该case原夹爪前80周期的完整因果历史已逐项重建，最大差为0，见[记录](../plans/v0.0.1/tracking/gripper/ordered_correct_history_rebuild_20261111.md)。
同一时刻机械臂两套响应历史与上一命令也已重建，最大差为0，见[记录](../plans/v0.0.1/tracking/joint_control/ordered_correct_arm_rebuild_20261111.md)。
统一质量机械臂与暂缓夹爪已在该正确参考上连续执行并独立重放16周期，位姿
16/16达门，但夹爪误差仍大，任务仅末周期成功；见[报告](../reports/tracking/joint_control/ordered_correct_stream_feedback_20261111.md)。

2026-11-11 模型安全有界回退：三个开发停止点由1、1、5周期延长至3、4、6周期，
但新增6周期均未达原位姿门、无抓取或任务成功；21条动作及525个内环样本独立
重放一致。详见[回退报告](../reports/tracking/joint_control/model_safe_tracking_fallback_20261111.md)。

2026-11-11 保持后的连续衔接：同一物理轨迹中衔接 8 周期，所选参考位姿 7/8、
最新参考 0/8、grasp 3/8，物体高度回到接近起点且无任务成功；24 条命令及
600 个内环样本独立重放一致。详见[衔接报告](../reports/tracking/joint_control/bounded_release_bridge_feedback_20261111.md)。

2026-11-11 有限释放保持物理对照：单个已研究状态的保持阶段接触 8/8、对所选
参考位姿 8/8，但对最新参考 0/8、无任务成功；400 个内环样本独立重放一致。
衔接及合法释放未执行，详见[反馈报告](../reports/tracking/joint_control/bounded_release_hold_feedback_20261111.md)。

2026-11-11 有限释放参考修订：18 条正确参考的 619 窗口无所定义冲突；12 条开发
记录有 20 个潜在冲突，但内环证据均未知，未激活保持。已研究闭合状态触发候选，
8 点旧 / 最新参考最大差 0.0513 m / 0.4397 rad，尚无新动作；详见[影子报告](../reports/tracking/joint_control/bounded_release_revision_shadow_20261111.md)。

2026-11-11 跨任务三级反馈：task0 固定新参考短窗实际 8/8 达双阈值，task1 执行
1/1 后无通过审核的下一候选；均无 grasp 或任务成功。9 条动作及 225 个内环样本
独立重放一致，详见[反馈报告](../reports/tracking/joint_control/replan_shared_shift_feedback_20261111.md)。

2026-11-11 共享预算三级反馈：raw实际1/1，terminal058由组合双点4/4延长至
5/5；新增一周期只通过首点审核，第二模型点超门。6条动作及150内环样本独立重放
一致，未完成任务或8周期，详见[反馈报告](../reports/tracking/joint_control/task2_shared_shift_feedback_20261111.md)。

2026-11-11 组合停止处后移计划：raw首点仍超原门，terminal058首点及完整模型
安全通过，但第二点超门；尚未派发或验证实际反馈。详见[资格报告](../reports/tracking/joint_control/task2_shared_shift_qualification_20261111.md)。

2026-11-11 共享预算组合实际反馈：raw恢复1周期1/1，terminal058保持余量分支
4周期4/4；独立重放5条动作及125内环样本一致，未完成任务或8周期。详见
[反馈报告](../reports/tracking/joint_control/task2_margin_guarded_feedback_20261111.md)。

2026-11-11 共享预算组合静态对照：task2 raw恢复原双点候选，terminal058保留
余量候选，均通过独立模型审核；另外两个任务起点仍无候选。尚无组合策略实际
反馈，详见[静态报告](../reports/tracking/joint_control/margin_guarded_fallback_20261111.md)。

2026-11-11 双点余量实际反馈：相同task2开发起点，raw首动作前停止，terminal058
实际4周期4/4达原误差门后停止；两者均未完成任务。独立重放一致，严格余量不宜
统一默认，详见[反馈报告](../reports/tracking/joint_control/task2_margin_two_step_feedback_20261111.md)。

2026-11-11 双点模型余量对照：相同四个预停止开发决策中，terminal058 step80有
0.0005 m / 0.003 rad模型余量的双点候选并通过独立审核；task2 raw和其余两决策
未找到。静态种子不等于在线细化种子，尚无新物理反馈；详见[对照报告](../reports/tracking/joint_control/margin_two_step_search_20261111.md)。

2026-11-11 terminal058 step85首点搜索：两种因果初值各耗尽320模型响应，均无
首点候选；最近安全尝试约0.010790 m / 0.057714 rad。原首点外界仍不确定，
有限搜索失败不证明不可达，详见[首点搜索报告](../reports/tracking/joint_control/task2_terminal85_first_search_20261111.md)。

2026-11-11 回退停止首点外界：raw step33在冻结模型首动作box/变化率放宽域的
联合下界为1.008605，排除原首点双阈值但余量很小；terminal058 step85的最强
下界为0.887498，尚不能判定。均不证明物理可达性，详见[外界报告](../reports/tracking/joint_control/task2_fallback_stop_pose_bound_20261111.md)。

2026-11-11 停止状态缓存审查：raw step33及terminal058 step85各用两个因果初值
重建双点搜索，291响应中的9个线性安全计划均无首点双阈值候选；未增加响应或新
物理。有限缓存结果不证明全局不可达，详见[审查报告](../reports/tracking/joint_control/task2_cached_first_pose_20261111.md)。

2026-11-11 因果后移回退：相同task2开发起点，raw仍1周期1/1后停止；terminal058
由原双点分支4周期4/4延长至5周期5/5，新增周期为独立审核的首点回退，第二模型点
超双阈值。6条动作及150内步独立重放一致，未完成任务或8周期，详见[反馈报告](../reports/tracking/joint_control/task2_causal_shift_fallback_20261111.md)。

2026-11-11 停止状态双种子搜索：raw step33和terminal058 step84各以已保存LP初值
及后移计划初值搜索，共346模型响应、0双点候选。terminal后移初值的最近安全尝试
第二点约0.010735 m / 0.054484 rad；不证明全局不可达，详见[搜索报告](../reports/tracking/joint_control/task2_shifted_seed_search_20261111.md)。

2026-11-11 停止状态后移计划：raw step33及terminal058 step84的后移计划均通过
冻结模型安全复核，但没有双点质量候选；terminal首点通过而第二点为0.013043 m /
0.066521 rad，raw前两点均超限。未重新优化或派发，详见[资格报告](../reports/tracking/joint_control/task2_shifted_seed_qualification_20261111.md)。

2026-11-11 双点计划模型偏差：重放过的5个开发周期显示，raw首周期后原下一动作
重锚预测为0.010372 m / 0.052042 rad，超原双阈值；terminal058最后周期后同一
原计划下一点为0.009502 m / 0.046570 rad，但再下一点尚未审核。关节位置/速度、
状态重锚和模型更新各有偏差；无新物理，详见[诊断报告](../reports/tracking/joint_control/task2_guarded_model_drift_20261111.md)。

2026-11-11 数值余量双点反馈：task2 raw由首动作前停止改善为实际1周期双阈值1/1，
随后无witness；terminal058仍为4周期4/4后停止，均未完成8周期或任务成功。
5条动作及125内步独立重放一致，104项CPU专项通过，详见[反馈报告](../reports/tracking/joint_control/task2_guarded_two_step_feedback_20261111.md)。

2026-11-11 双点搜索数值尺度对照：同一task2 raw静态输入从无witness变为独立
审核通过的双点模型候选，terminal058仍有候选；task0/1仍无witness。最终第二点
贴近原阈值，尚无新物理反馈；32项CPU专项通过，详见[对照报告](../reports/tracking/joint_control/guarded_two_step_search_20261111.md)。

2026-11-11 双点硬约束实际反馈：task2 raw首动作前无witness，terminal058执行
4周期且实际双阈值4/4、关节缓冲未失守，随后无witness；未完成8周期或任务成功。
4条动作及100内步独立重放一致，91项CPU专项通过，详见[反馈报告](../reports/tracking/joint_control/task2_two_step_feedback_20261111.md)。

2026-11-11 提前一拍双点搜索：四个相关开发状态中，task2 terminal058找到独立
审核通过的前两点模型witness，后六点仍有质量缺口；另三处无witness，raw第二点
仅数值边界超限仍被拒绝。228响应、29项CPU专项通过，尚无新物理反馈，详见
[双点报告](../reports/tracking/joint_control/two_step_viability_20261111.md)。

2026-11-11 三任务停止位姿外界：四个已重放开发停止状态在冻结模型首动作box/变化率
放宽域内均排除位置和姿态同时达标；task0/1为联合冲突，task2 raw/terminal分别
由位置/姿态单项排除。40次LP及4个独立标量响应复核通过，30项CPU专项通过；这不
证明真实物理不可达，详见[联合外界报告](../reports/tracking/joint_control/stopped_pose_safety_bound_20261111.md)。

2026-11-11 两阶段首点优先：两个task2开发变体各执行1周期后仍无witness；实际
双阈值分别0/1和1/1。第二次搜索分别评估21/12个线性安全候选但无首点合格解；
2条动作及50内步独立重放一致，84项CPU专项通过，详见[反馈报告](../reports/tracking/joint_control/task2_staged_first_feasibility_20261111.md)。

2026-11-11 可行初值恢复：两个task2分支各执行1周期后仍无witness；第二次决策
分别评估到6/1个线性安全候选，但均未满足首点质量。2条动作及50内步独立重放
一致，73项CPU专项通过，详见[反馈报告](../reports/tracking/joint_control/task2_feasible_start_restoration_20261111.md)。

2026-11-11 首点硬恢复：task2两个开发变体各执行1周期后在下一次320响应预算内
无witness；raw实际首点位置略超阈值，terminal首点双阈值通过。两次停止搜索均
未评估到线性关节安全候选，2条动作及50内步独立重放一致；76项CPU专项通过，
详见[反馈报告](../reports/tracking/joint_control/task2_first_hard_restoration_20261111.md)。

2026-11-11 task2多点恢复：相同step32/80起点分别执行0/1周期后，候选硬约束审核
拒绝，未改善此前首点质量分支的持续性；唯一动作及25内步独立重放一致。69项CPU
专项通过，0新WAM/GPU/fresh，详见[恢复报告](../reports/tracking/joint_control/task2_earlier_restoration_feedback_20261111.md)。

2026-11-11 task2提前反馈：在参考生成的step32/80，原细化、首点质量、八点质量
两组分别执行8/1/0和8/5/0周期；没有完整八周期同时满足原位姿和关节缓冲，四处
质量分支因无witness停止。terminal原细化末周期有11个内环样本跌破缓冲；22条
动作和550内步独立重放一致，60项CPU专项通过，详见[反馈报告](../reports/tracking/joint_control/task2_earlier_preview_feedback_20261111.md)。

2026-11-11 首周期位置外界：task2 raw step34在冻结模型中，仅保留首动作box和
变化率时的位置误差下界约0.017674 m，超过原0.01 m阈值；另外三个状态的外界
不足以判定。256个box角点独立复算通过，31项CPU专项通过；这不是物理不可达
证明，详见[外界报告](../reports/tracking/joint_control/first_step_position_bound_20261111.md)。

2026-11-11 三任务可行性优先对照：四个开发状态的原/新搜索均只在同一task2预警
起点找到独立审核通过的模型witness；另三个状态均无witness，改变搜索顺序没有
新增候选。8次求解共752响应，0物理积分/WAM/GPU/fresh，58项CPU专项通过；
有限搜索失败不证明不可达，详见[对照报告](../reports/tracking/joint_control/multitask_feasibility_first_20261111.md)。

2026-11-11 task2早期预警反馈：两个开发变体的原/质量分支分别执行8/0及6/3周期；
两处质量分支无witness，原细化一处实际11个2 ms样本跌破0.05 rad缓冲并在下一
动作前停止。四分支均无任务成功，17条动作及425内步独立重放逐值一致；35项CPU
专项通过，无新WAM/GPU/fresh，详见[反馈报告](../reports/tracking/joint_control/task2_warning_feedback_20261111.md)。

2026-11-11 task2早期预警前缀：两条seed43开发记录分别重放34/82周期，118个端点
状态逐字节一致，末周期2 ms内环均守0.05 rad缓冲；共116周期、2900内步，0新
选择动作/WAM/GPU/fresh。26项CPU专项通过；这不是安全控制收益，详见
[前缀报告](../reports/tracking/joint_control/task2_warning_prefix_20261111.md)。

2026-11-11 打开事件对照：四处影子机会中两处当前关节已越界而未准入；另两处的
旧/新参考四分支分别执行8、1、8、8周期，旧参考一处第2步前无witness。task1旧
参考打开更早，但对最新TCP仅4/8双阈值；四分支均无grasp或任务成功。25条动作及
625个内步独立重放逐值一致，49项CPU专项通过；无新WAM/GPU/fresh，详见
[对照报告](../reports/tracking/joint_control/causal_opening_event_feedback_20261111.md)。

2026-11-11 批量质量求解对照：三个相关开发时刻各两次原/批量求解均获witness，
12次独立审核通过，成对命令最大差6.73e-14；批量求解约0.887–0.927秒，仍不具
20 Hz资格。默认控制器未切换，0新物理/WAM/GPU/fresh；68项CPU专项通过，详见
[对照报告](../reports/tracking/joint_control/batched_quality_solver_20261111.md)。

2026-11-11 批量求解器对照首次QA在收集前拒绝：原求解器/历史测试的SHA与迁移
身份清单冲突，未执行测试或求解；原字节已恢复并核验，后续使用独立候选入口，
详见[QA记录](../reports/tracking/joint_control/batched_quality_solver_qa_incident_20261111.md)。

2026-11-11 批量自由预览：三个已保存状态共153对展开逐字段通过1e-10容差，
最大差7.11e-15；原/批量中位耗时约4.40–4.46/2.48–2.51 ms，批量准备约0.27 ms。
63项CPU专项通过；尚未接入求解器或物理控制，也不具20 Hz资格，详见[对照报告](../reports/tracking/joint_control/batched_joint_rollout_20261111.md)。

2026-11-11 质量求解耗时剖析：三个已保存开发状态各两次结果完全一致，原独立
审核均通过；不带profiler求解墙钟1.07–1.12秒，审核约21 ms。SLSQP与关节轨迹
展开各占profiler self time约两成；零内环误差仅作离线诊断，不授予安全/实时
资格。55项CPU专项通过，0新物理/WAM/GPU/fresh，详见[耗时报告](../reports/tracking/joint_control/quality_solver_runtime_profile_20261111.md)。

2026-11-11 两因素实际对照：两种位姿下，旧opening的切换后8/8端点grasp，衔接
opening为3/8；旧位姿+旧opening末端抬升5.31 cm，而衔接位姿+旧opening仅0.16 cm。
四组任务成功均为0；旧opening明显偏离最新目标。32条新动作独立重放逐值一致，
182项CPU专项通过；另保留原中断的未知额外费用，详见[对照报告](../reports/tracking/joint_control/causal_event_bridge_factorial_20261111.md)。

2026-11-11 两因素对照中断：两路producer最后各记录64周期原前缀，进程和句柄已
消失，无退出或完整结果收据；实际总费用未知。原部分证据保持，另行登记从原初态
的独占恢复，详见[中断账本](../reports/tracking/joint_control/causal_event_factorial_interruption_20261111.md)。

2026-11-10 因果衔接实际反馈：task1 seed42旧参考8周期后，经固定8周期过渡至
已生成新参考，16条动作均执行；实际TCP对衔接参考16/16双阈值、对最新参考7/16。
grasp端点367–371，物体最高较起点升高1.38 cm，最终回到起点附近，任务成功0；
opening仍有较大误差。16动作独立物理重放逐值一致，151项CPU专项通过；固定参考
短窗、非实时且0新WAM/GPU/fresh，详见[衔接报告](../reports/tracking/joint_control/causal_event_bridge_feedback_20261110.md)。

2026-11-10 闭合后切换参考：task1 seed42原有8周期重放一致，端点368在完整保留
控制历史后切回step360生成的新参考；首个质量搜索无witness，第9条动作未派发，
夹爪提案撤销。前8动作独立物理重放逐值一致；112项CPU专项通过，0新WAM/GPU/fresh。
有限搜索失败不证明全局不可达；详见[切换报告](../reports/tracking/joint_control/causal_event_switch_feedback_20261110.md)。

2026-11-10 闭合记忆16周期实际对照：task1 seed42旧参考分支连续10个端点grasp、
物体较起点升高5.31 cm，新参考分支无grasp；两分支均无任务成功。旧分支对最新
参考仅5/16双阈值，最大偏差0.05161 m/0.42569 rad。32条动作及物体位置独立
重放逐值一致，103项CPU专项通过；固定参考跨原重规划点、非实时、零新WAM/GPU/fresh，
详见[报告](../reports/tracking/joint_control/causal_event_long_feedback_20261110.md)。

2026-11-10 闭合记忆匹配物理对照：两案例各新/旧参考共4分支；task1旧参考8周期
中2个grasp端点、对最新参考5/8双阈值，新参考无grasp且自身8/8；两者均无任务
成功。task0旧参考首动作前无witness停止，新参考自身4/8。已执行24动作的内环
均守0.05 rad缓冲、独立物理重放逐值一致；125项CPU专项通过。固定参考短窗且非
实时，0新WAM/GPU/fresh；详见[反馈报告](../reports/tracking/joint_control/causal_event_feedback_20261110.md)。

2026-11-10 因果夹爪历史重建：两条seed42前缀的496个夹爪决策、预览、滞回及
规划输出均与原日志逐项一致，最大差0；496次planner共3254256条候选序列。
两起点前一命令相同但intent方向不同；115项CPU专项通过，零新物理/WAM/GPU/fresh。
这是共享实现的历史重建，不是新策略成功，见[夹爪报告](../reports/tracking/joint_control/causal_gripper_history_20261110.md)。

2026-11-10 闭合机会的原生命令前缀：seed42两案例各自重放136/360周期，498个
端点状态逐字节一致，已保存控制与夹爪缓存最大差0；496周期/12400个2 ms内步，
无新候选动作/WAM/GPU/fresh。80项CPU专项通过；只能证明两条前缀在当前运行时
可复现，不能证明候选成功，详见[前缀报告](../reports/tracking/joint_control/causal_event_prefix_replay_20261110.md)。

2026-11-10 因果事件记忆影子回放：12条开发轨迹900次查询中6次选择旧参考，分别
是6案例的首次改变；2次闭合、4次打开，其中1次引用两次重规划前的闭合事件。
其后235次仍使用原基线观测，不能当新策略轨迹。独立逐项复算的最大差1.11e-16；
63/77项CPU专项通过，零新物理/WAM/GPU/fresh。是否可安全执行和改善任务仍未验证，
见[影子报告](../reports/tracking/joint_control/causal_event_memory_shadow_20261110.md)。

2026-11-10 开发任务阶段重规划分析：12条原轨迹、888对chunk及28416个同绝对
端点已独立复核；位置/姿态差中位0.02743 m/0.11527 rad。旧首闭合预测在执行窗外
的136对中，新预测87次推迟、29次未见；一条案例75个chunk均无闭合预测。探索性
下一8步保留交集仅闭合1次、打开4次，非反事实成功率。两轮专项43/51项通过，
0新物理/WAM/GPU/fresh，详见[阶段报告](../reports/tracking/joint_control/replan_task_event_stability_20261110.md)。

2026-11-10 新参考质量恢复反馈：task0实际7/8，低于八点硬质量的8/8；task1
执行1周期后审核拒绝，与首点质量分支推进长度相同。已执行内环守住0.05 rad
缓冲，9动作及225内步独立重放逐值一致；70项CPU专项通过，0新WAM/GPU/fresh。
不能据此提升为统一候选，详见[反馈报告](../reports/tracking/joint_control/replan_restoration_feedback_20261110.md)。

2026-11-10 新参考八点质量实际反馈：task0完成8周期、实际双阈值8/8，原反馈6/8、
首点质量7/8；task1在首条新动作前未找到八点witness并停止。已执行内环守住0.05 rad
缓冲，8动作及200内步独立重放逐值一致；决策1.294–1.491秒/周期不具实时资格。
61项CPU专项通过，0新WAM/GPU/fresh，详见[反馈报告](../reports/tracking/joint_control/replan_full_horizon_feedback_20261110.md)。

2026-11-10 新参考重规划八点模型搜索：task0两种初值均找到经独立复算的八点
模型witness；task1均未找到，最优已审核关节候选的最差归一化质量超限0.5611。
有限搜索不证明全局无解，尚未派发；27项CPU专项通过，0物理/WAM/GPU/fresh，
详见[模型检查报告](../reports/tracking/joint_control/replan_full_horizon_quality_20261110.md)。

2026-11-10 新参考质量反馈：task0质量分支实际7/8双阈值（原反馈6/8），仍有
26微米位置超限；task1质量分支只执行1周期，第2次决策前无witness并停止。
已执行内环守住0.05 rad缓冲，9动作独立重放逐值一致；44项CPU专项通过，
0新WAM/GPU/fresh，详见[反馈报告](../reports/tracking/joint_control/replan_quality_feedback_20261110.md)。

2026-11-10 同状态新旧参考实际对照：两案例均从原始端点16开始，跟踪new参考
自身双阈值分别6/8、3/8；跟踪old参考自身均8/8，但对共同new参考均只有1/8。
四分支关节内环守住0.05 rad缓冲，独立重放32动作逐值一致；47项CPU专项通过，
0新WAM/GPU/fresh，详见[对照报告](../reports/tracking/joint_control/replan_reference_switch_20261110.md)。

2026-11-10 组合预警16周期实际反馈：两案例的原反馈和质量反馈均完成16周期、
实际双阈值16/16且内环守住0.05 rad缓冲；原反馈同样达标，未证明质量约束收益。
64条动作及内环独立重放逐值一致，首8周期与上一轮一致；62项CPU专项通过，
0新WAM/GPU/fresh。固定step8参考跨两个原重规划时刻，非新WAM闭环；见[报告](../reports/tracking/joint_control/combined_warning_long_feedback_20261110.md)。

2026-11-10 组合首次预警实际反馈：两案例的原反馈和质量反馈均完成8周期、
实际双阈值8/8且2 ms关节内环守住0.05 rad缓冲；原反馈同样达标，未证明质量
约束有收益。独立重放32条动作逐值一致；45项CPU专项通过，0新WAM/GPU/fresh。
窗口结束于原首次超限之前，且跨step16使用固定旧chunk后缀，详见[反馈报告](../reports/tracking/joint_control/combined_warning_feedback_20261110.md)。

2026-11-10 组合质量与速度预警回顾：12条原开发记录中11条曾出现位姿超限，
组合规则在11条的首次超限前报警，原速度规则仅1条提前；另1条组合报警但至
端点599未观察到位姿超限。结果是已研究数据的回顾，不代表可恢复控制或任务成功；
28项32并发CPU专项通过，0物理/WAM/GPU/fresh，详见[报告](../reports/tracking/joint_control/combined_quality_warning_20261110.md)。

2026-11-10 额外模型制动静态消融：保持0.05 rad模型内环缓冲及原位姿阈值，
去掉额外q+0.1v约束后，两案例最差归一化质量超限从0.60770/2.86481小幅降至
0.56787/2.73899，均无8点质量witness，且候选违反额外制动预测；47项CPU专项
通过，0物理/WAM/GPU/fresh，详见[消融报告](../reports/tracking/joint_control/braking_constraint_ablation_20261110.md)。

2026-11-10 更早预警的多点可行性恢复：两案例关节安全模型候选的最佳最差
位姿归一化超限为0.60770/2.86481，均未找到全8点质量witness；0.1 s模型
制动上限在最佳候选处近活跃。42项CPU专项通过，0物理/WAM/GPU/fresh，
模型局部优化不证明全局无解，详见[恢复报告](../reports/tracking/joint_control/full_horizon_feasibility_restoration_20261110.md)。

2026-11-10 更早速度预警多点搜索：task0端点39首点模型解复算通过，但两次
全8点搜索未找到witness；task1端点47在警报前已姿态超限，首点及全8点均未
找到witness。两处初始计划预测分别5/8、0/8达标，不同时间参考chunk不同；
36项CPU专项通过，0物理/WAM/GPU/fresh，见[报告](../reports/tracking/joint_control/velocity_warning_full_horizon_20261110.md)。

2026-11-10 首次预警起点的8点质量模型搜索：两案例、原计划和单点质量热启动
四次搜索均未找到全8点witness；线性关节约束可行，非线性线搜索终止。原种子
模型各4/8点满足双阈值；0物理积分/WAM/GPU/fresh。有限搜索不证明全局无解，
34项CPU专项通过，见[多点搜索报告](../reports/tracking/joint_control/full_horizon_quality_feasibility_20261110.md)。

2026-11-10 质量反馈停止状态的冻结模型诊断：两处均找到满足关节约束的模型
计划，但SO(3)首点误差分别0.073236/0.064913 rad，均超0.05 rad；有无经验
收紧及关节witness热启动的六次质量搜索均达到60次迭代而未找到解，不能证明
全局不可达。43项CPU专项通过，0物理积分/WAM/GPU/fresh；详见[诊断报告](../reports/tracking/joint_control/quality_stop_feasibility_20261110.md)。

2026-11-10 两案例质量约束实际反馈：质量分支各执行5周期后均因有限搜索
未找到下一witness而在派发前停止，已执行内环守住0.05 rad缓冲；位姿双阈值
分别5/5和4/5，均未完成8周期。task1模型姿态卡在0.05 rad边界，实际略超；
26个实际新动作独立重放一致，56项CPU专项通过，0新WAM/GPU/fresh。有限
搜索失败不证明全局不可达，详见[反馈报告](../reports/tracking/joint_control/early_quality_feedback_20261110.md)。

2026-11-10 两案例16状态质量模型检查：绝对首点位姿约束在每条轨迹前5个
状态找到模型witness、后3个状态未找到；过去误差收紧后找到与否相同。18个不同
witness的独立模型关节复算通过，尚未实际派发；原单周期模型与实际旋转误差
向量最大差约0.000504 rad。35项CPU专项通过，0物理/WAM/GPU/fresh，见
[质量检查报告](../reports/tracking/joint_control/early_quality_feasibility_20261110.md)。

2026-11-10 首次预警原细化器闭环：两案例各8周期的实际2 ms内环均守住原
0.05 rad关节缓冲，无越界；但各自后4点旋转误差超0.05 rad，位姿双阈值仅
4/8，未获完整跟踪资格。两类历史由各分支实际完成端点更新，独立动作重放
一致；两阶段共444控制周期/11100内步，44项CPU专项通过，0 WAM/GPU/fresh。
详见[闭环报告](../reports/tracking/joint_control/early_refiner_feedback_20261110.md)。

2026-11-10 首次预警原约束细化重建：从两条原初态重放至端点43/52，依据上一
完成周期预测及当前实测端点重算两类创新；两案例原细化动作最大差分别
2.81e-13/3.13e-14，完整计划及route亦复现。本次未派发新动作；31项CPU专项
通过、95原命令周期、0 WAM/GPU/fresh，见[重建报告](../reports/tracking/joint_control/early_refiner_reconstruction_20261110.md)。

2026-11-10 两案例首次预警实际对照：历史8周期内环仍守住0.05 rad缓冲；
四个新预览/退避分支在执行4–5周期后均实际越界并停止，虽已执行的19个周期
位姿双阈值全部达标，仍全部不具安全资格。六分支动作独立重放一致，合计
640控制周期/16000内步，55项CPU专项通过，0 WAM/GPU/fresh；详见
[实际对照](../reports/tracking/joint_control/early_warning_tracking_20261110.md)。

2026-11-10 12案例因果预警回顾：固定0.10 rad及当前速度0.2 s外推在原7条
端点越界案例中均曾于0.05 rad缓冲内提前报警；原5条未见端点越界的案例中
各有4条报警，且不少报警后的四个端点未越界。两案例固定0.10首次警报为端点
43/52，前一周期跟踪达标；尚未测试该触发后的控制。31项CPU专项通过，零新
物理/模型/GPU/fresh，详见[预警回顾](../reports/tracking/joint_control/causal_warning_audit_20261110.md)。

2026-11-10 两案例短时安全对照：历史分支在8周期内分别有16/120个2 ms样本
实际关节越界，限速归零及Jacobian退避的四条分支均无实际越界；但四条都低于
0.05 rad原安全缓冲、位姿双阈值均为0/8，未获控制资格。六分支独立运行时按
保存动作重放一致；两阶段共1374控制周期/34350积分步，66项CPU专项通过，
0新WAM/GPU/fresh。起点后验选择，非在线触发或任务恢复证明；见
[安全对照报告](../reports/tracking/joint_control/multicase_emergency_retreat_20261110.md)。

2026-11-10 两条开发轨迹前缀重放：task0/task1各一条从原初态重放114/99个
保存命令，215个端点的完整state逐字节一致，已保存控制器缓存最大差0；两子
进程正常回收。新增213个仿真控制周期、5325个积分步，60项CPU专项通过，
0新WAM/GPU/fresh。历史initial_joint未单独保存；未测试新安全动作，详见
[重放报告](../reports/tracking/joint_control/multicase_native_prefix_replay_20261110.md)。

2026-11-10 多案例越界上下文：7条首次越界均在关节上限，5条涉及q7、各1条
涉及q5/q6；丢失该关节0.05rad缓冲到越界相隔3–16周期。首次越界端点仅1条
观察到机器人—场景接触，不能据此排除未采样时刻的接触或归因。34项专项通过，
新增28次静态forward、无时间推进/WAM/GPU/fresh；见[报告](../reports/tracking/joint_control/multicase_joint_limit_context_20261110.md)。

2026-11-10 多案例开发诊断闭合：原首轮6条完成、6条失败的记录保持，另行
补齐6条并通过33项CPU专项。12条既有开发轨迹中7条存在实际仿真关节越界；
11条无目标grasp接触，另1条有接触但任务仍失败。424个含预测闭合的查询中，
首个闭合目标对冻结对象仅7个有双pad静态接触；该数字不能当动态成功率。
未新增时间推进/WAM/GPU/fresh；见[多案例报告](../reports/tracking/joint_control/multicase_reference_geometry_20261110.md)。

2026-11-10 多案例诊断首轮：12条已有开发案例中6条完成、6条因新增对象旋转
逐位一致检查失败（最大差约4.4e-16至6.7e-16）；原总结果保持失败，12个
worker实际回收。对象状态上的浮点重复归一化在合成例可复现同量级差异，不能
据此改判原结果。已单独登记仅补齐6条缺失案例，控制和任务门槛不变；见
[失败报告](../reports/tracking/joint_control/multicase_reference_geometry_v1_20261110.md)。

2026-11-10 完整轨迹细化：共同安全基线分别在原 root 域和完整输入域进行
有界非线性优化，两组均触及8次迭代上限；所有响应独立逐条复算一致，完整
witness均为0。两组最优合格结果相同，仍为12/24 caps、prefix2，改善很小；
求解器返回值未通过模型安全及逐项保护。62项专项通过，新增物理/WAM/GPU/
fresh均为0。按预登记转向多开发参考、上游质量和实际安全分析；见
[完整报告](../reports/tracking/trajectory_search/whole_trajectory_refinement_20261110.md)。

2026-11-10：中断投影保存的236条响应已做后验自洽筛查；13条标记安全的候选
及不安全对照/重复共15响应已独立native复算、逐值一致。13条虽满足原安全
buffer，却均只有10/24 caps、prefix2；最佳最差误差的候选终点位置49.85mm，
基线为16.90mm，没有完整witness。原实验最终账本和中断时部分费用仍未知，
不追认为完整通过；其余223条未独立复算。见[复核报告](../reports/tracking/trajectory_search/interrupted_projection_assessment_20261110.md)。

2026-11-02算法域核查：保留原安全/变化率行，仅诊断完整input域时，终点支持
下界由原root约1.45变为约-6.58。旧局部排除不能升级为机器人不可达，负支持
也不证明完整caps可行。29项专项及2LP审核通过，0新响应/物理/GPU。
已重新审查旧4000响应非线性搜索的失败，后续将设计不同起点的非线性可行性
恢复，避免简单重跑或只扩大线性域；见[域诊断报告](../reports/tracking/trajectory_search/input_domain_support_20261102.md)。

2026-11-02固定终点方向核查已完成：10双侧响应逐值exact，61项专项通过。
三种步长终点旋转改善但仍超cap，实际reserve分别-171.8/-39.7/-8.2，全部joint
audit失败；预测却保有约.9，说明较大步的非线性安全误差不能忽略。没有候选
下发、没有新物理/GPU。后续需区分算法域限制与非线性安全模型瓶颈，不能把
局部下界升级为物理不可达；见[完整报告](../reports/tracking/trajectory_search/local_support_direction_probe_20261102.md)。

2026-11-02完整瓶颈诊断：48个逐cap乐观LP及原单位对偶检查通过；初始和最终
局部模型的终点旋转支持下界，在原实际buffer下仍为1.445/1.452（达标须≤1），
reserve .9仅改变约2.1e-4。提示需审查终点方向与非线性模型可信度，而非继续
降低进展门。LP步长接近.2，不能把局部模型下界当作物理不可达证明。
38项专项通过，0新响应/物理/GPU；见[完整诊断](../reports/tracking/trajectory_search/local_tracking_support_20261102.md)。

2026-11-02最新：小进展阈值组的保留36响应已另行独立重放通过，原审核接线
失败结果保持False。新组12stage/7接纳，仍caps12/24、prefix2、witness0，
最终loss2.008885略高于历史安全反馈2.008818。按预登记停止继续调整标量
进展阈值，转向完整轨迹误差、约束方向与可执行性瓶颈审查。94项专项通过，
无GPU或物理执行；总目标与完整验收均未完成。
见[实际结果与独立补齐报告](../reports/tracking/trajectory_search/small_progress_search_20261102.md)。

## 总目标

在固定机器人和控制接口下，将世界模型给出的期望状态轨迹，结合当前实测状态
与因果历史，转换为可执行action，使实际状态可靠跟踪目标，并在留出新初态上
稳定完成任务。主要指标是经过固定前向控制系统后的状态、安全及任务稳定性。

原18正确参考/12 WAM开发/24最终留出门槛保持。最终稳定需≥20/24且各任务≥4/8；
最终留出尚未打开。当前总目标未达成，局部数值与测试通过不能代替该验收。

## 仓库整理：目录迁移与分轮验收

[结构设计](architecture/repository_design.md)定义层次、目标目录、身份迁移协议、
分阶段提交及验收。先建立来源清单与路径基础，再迁移trajectory-search完整切片，
随后按依赖闭包整理其他领域。算法、控制策略、数值容差和验收门槛保持。

所有文件及产物在当前仓库维护；根目录用户未跟踪文件不纳入整理。
生成的outputs、现有环境、XML和运行证据本次保持原位置。GPU继续禁用。

已完成设计提交`0c5ed51`和3276文件基线；D1路径/来源身份基础已实现，36项CPU
专项及真实catalog接线通过。迁移前测试发现4976节点，5项既有PyArrow/NumPy
导入错误已单列。[迁移记录](migrations/README.md)说明证据和接口边界。

D2首批85项已按主题移动，配置/计划/报告原字节保持，8算法模块的38个顶层
函数/类AST保持；迁移后182个原QA节点完整保留并通过。根测试辅助导入统一包
身份。最终302项统一CPU QA、153个Python文件Ruff/格式及两真实输入dry通过；
旧4976测试节点完整保留、新增120，仅原5项环境collection错误。结构/身份检查
通过，D3其余领域尚待迁移；未重跑生产或升级原实验资格。

D3a已把其余377个库模块迁入领域包，src/s2a根目录只保留包入口。库层反向
依赖scripts的两处函数闭包已提取；当前脚本统一采用scripts.*导入身份。全量发现
初次5169节点、修复新增拒绝边界后最终5171节点；原5096节点的函数、参数和顺序
完整保留，仍只有原5项
PyArrow/NumPy环境错误。32进程674项CPU回归用时约49秒，首次669通过、5项
旧路径/导入匹配失败按原记录保留；修复5项独立并行复核全部通过，另两相关模块
21项通过（包含新增2项拒绝边界测试）。两真实dry通过，
QP/响应/查询为0。脚本、测试及历史证据的其余物理迁移仍待D3b–D4执行。

D3b首批775个共享/审核脚本已按领域移动，1141个当前调用方同步明确包路径；
独立metadata审核仍保持仅标准库，原数学校验函数AST保持。累计清单登记3280个
历史身份，其中1237个已移动。全量发现5172节点、原节点及参数保持，仅原5项
环境错误。2254项初测2107通过、147项旧路径等迁移适配失败；修复后35个完整
相关模块486项复核482通过，最后4项独立复核全部通过。全仓2470个Python文件
Ruff/format通过。上述为分轮验证，不冒称完整pytest或同版本全量重跑已通过。

D3b其余690个脚本已按设计移动，57个测试调用方明确当前包和存储位置，新增17个
包标记；scripts根目录只保留README和包入口。累计3280个历史身份中1927个已
移动。全量发现仍为5172节点，原节点/参数/顺序保持；全仓2487个Python文件
Ruff/format及静态导入边界检查通过。816项32进程CPU初测781通过、34失败、1项
按原条件跳过；修复后11个完整模块148项复核139通过、9失败，随后9项复核8通过，
最后1项单独修复复核通过。原失败收据均保留，未称最终版本全部816项重跑通过。

PyArrow相关5个模块另在仓库现有Python3.12/NumPy2环境中执行14项，全部通过；
3.11数值环境的原5项collection错误仍保留，未安装或修改共享环境。历史CLI帮助
与发布桥接仅在原字节的隔离子进程中回归，禁止模型/数组读取及混入当前源码，
不授予当前运行或控制资格。

D3c已实际迁移439个测试文件，新增71个真实包标记，测试根只保留README、
__init__和原字节conftest。pytest保留原scope，仅移除旧scripts搜索路径；并发CPU
入口已写入`scripts/maintenance/run_cpu_tests.py`。原5172节点完整保留，新增5项
原生运行时来源篡改拒绝用例。全仓2559个Python文件Ruff/format通过；本阶段迁移
回归已分轮收尾，原环境限制与验证版本分别保留如下。

首次1205项回归1141通过、64失败。分工修复来源/CLI定位后，首次64进程全量
5177项报告4987通过、188失败、2跳过，另有原5项环境collection错误；该轮还
暴露迁移时遗漏的模块替身导致一份真实config被合成fixture覆盖，source_unchanged
为False，因此整轮不作为有效通过记录。全部进程已回收，原config已从Git精确
恢复并核原SHA，完整3280项复核没有其他漂移。测试替身和临时写入边界已修复，
其余历史loader/独立来源证明及数值环境差异继续分别处理，原失败收据保留。

其中streaming的4项严格数值失败已用迁移前原test及原数学库，在隔离3.11环境中
实际复现：原26项22通过、同4项失败，误差逐项相同；当前开发环境原容差下26项
通过。两环境的Python/NumPy/OpenBLAS均不同，不能单独归因其中一个版本。本次
不修改算法或容差，后续全量记录须分开这些既有数值限制与新增迁移失败。

修复后的64进程全量5177项报告5169通过、6失败、2跳过，来源前后相同；6项为
上述4项既有数值差异和最后2项迁移适配失败。随后仅修改这两个测试，完整相关
模块51项并发复核全部通过，没有声称最终版本重新执行全部5177项。兼容开发
环境另有Arrow14项与streaming26项共40项全部通过，来源冻结且实际回收。当前
没有未闭合的已知新增迁移失败，但不能称单一环境完整pytest全绿。源码身份与
结构检查通过，累计2366个原文件已移动；完整raw/nav迁移和最终D5仍待执行。

D4已实际移动其余898份配置/计划/报告，全部928份原始文件的字节、大小和Git
blob身份保持。README收敛为目标与导航，三份基线README原文另行归档；AGENTS
只更新001目标导航并追加CPU偏好，原约束保持。25个测试模块只适配当前I/O边界，
Ruff/format通过，算法和raw JSON值保持。累计3264个原文件已移动，根平铺为
src1/scripts2/tests3/configs1/plans1/reports1。D4提交时最终D5尚待执行；以下记录
本次提交前的D5验收，研究目标仍未达成。

D5首次64进程全量发现5177节点，5138通过、37失败、2跳过，另有原5项环境
collection错误；37项中33项为深层raw读取迁移适配，4项为已复现的原数值差异。
修复后全量v2为5170通过、5失败、2跳过，源码前后相同；其中仅1项仍为遗漏的
合成fixture PLAN读取，现已补齐。独立审查另补齐两个preview真实BASE_REF读取，
显式传入各自引用并严格核原声明/catalog/current SHA，替换引用不回退加载基线。

最终冻结版本的14个完整相关模块179项全部通过，覆盖原33项迁移失败；兼容环境
40项全部通过。重新发现5177节点及原顺序不变，仅保留原5项环境collection错误。
两条最新真实入口dry、完整结构及静态导入检查通过；四个最终运行owner的3646项
源码/输入清单相同，验证期间无漂移且所有子进程实际回收。全仓2559个Python文件
Ruff/format通过，928份raw再次核原字节一致，累计3264个实际移动符合设计。

最后修复后没有再完整执行全部5177项，不能把上述分轮结果称为单一版本完整
pytest全绿。目录、身份及本轮读取回归验收不授予所有历史入口执行资格；原控制
验收、失败证据和GPU禁令保持。当前累计清单SHA及各轮收据见
[D5迁移记录](migrations/README.md#d5最终结构与读取边界验收)。

## 整理前的控制与实验状态

- 已有明确DROID adapter与LIBERO OSC接口、因果参考/历史、前向预测及独立物理
  评估链路；各自证据等级和控制语义分别记录。
- 当前实际资格仍cursor8/task0/grasp0，尚无新的完整H12执行轨迹资格。
- 最新dual-recovery唯一生产已正常关闭：122响应、52QP、52次拟合；51QP通过
  原证书，完整数值witness仍0。保存数据表明总误差改善主要来自已达标前九点，
  后三点仍超原caps，不能据此宣称可靠跟踪。
- 首次checker在来源读取阶段失败，原False与零响应失败记录保持。隔离check-only
  v2实际复算了全部122响应/36600内步/38065查询，原完整KKT账本通过，但最终
  fresh_ratio差2.137171550842254e-10超过原1e-10，complete/passed仍False。
- 整理后仍须解决严格决策核验及轨迹目标分配问题，再按独立登记继续控制研究。
  不因为搬目录追认失败、放宽容差、追加旧预算或自动恢复模型任务。

最新证据路径与SHA见[设计中的迁移前快照](architecture/repository_design.md#8-迁移前实验状态快照)。

## 验证边界

2026-11-02：新增尚未接入搜索的独立 OSC 算术实现。原生 NumPy 环境下 129 个
输入与独立回放函数逐字节一致，三个专项测试通过；原预测器及容差未改。
这只证明力矩代数一致性，完整响应、严格 ratio 及任务资格尚未通过。
证据和下一步见[算术契约报告](../reports/tracking/trajectory_search/native_osc_arithmetic_20261102.md)。

后续固定命令完整响应对照：两侧共8条H12响应通过原响应容差，但并非完全同值。
首次失败命令对的条件性 ratio 差降至5.18e-11，尚未授予完整搜索资格；另发现
Jacobian存储布局影响，下一步核查dot/@路径。失败、额外成本和5项专项测试见
[完整响应报告](../reports/tracking/trajectory_search/native_replay_response_contract_20261102.md)。

最新v4已在固定四条命令上得到与独立回放完全同值的完整H12响应：修正任务
dot/@并在采样时保留J布局，响应及独立指标最大差0，13项专项通过。
来源、实际成本及边界见[布局契约报告](../reports/tracking/trajectory_search/native_replay_layout_contract_20261102.md)。

新预测器初始化及72个signed1e-4探针现已重新验证：两侧各74完整H12响应、
评分和Js/Jq/Je矩阵精确同值，因果零结构通过。生产端32批并发，全部进程实际
回收；23项专项通过。原18/24初始化caps尚未全部满足；证据与成本见
[新探针报告](../reports/tracking/trajectory_search/replay_order_stencil_20261102.md)。

同预测器sum-q对照已完成：122生产响应及全部独立复算、完整KKT/反馈/原1e-10
fresh决策审核通过，响应同值。总q29.62→27.47，但后3点18.80→18.96，仍18/24
caps、prefix9、witness0；未获得物理资格。61项专项通过，下一步按设计执行
同预测器cap-distance对照。费用与边界见
[sum-q报告](../reports/tracking/trajectory_search/replay_order_sumq_20261102.md)。

Cap-distance目标与显式objective共享驱动已实现，58项专项通过；共享驱动的
sum-q路径与闭合原基线完整软件回放同值，没有新增研究运行。浮点边界的有理化
求值已另行固定。该实现阶段见
[实现报告](../reports/tracking/trajectory_search/cap_distance_implementation_20261102.md)。

独立cap审核及唯一对照现已完成：79项专项通过，89生产响应和全部独立复算、
原KKT/反馈/严格fresh及owner源码/回收审核通过。loss2.231→2.010、后3点改善，
但caps18/24→12/24、prefix9→2、witness0，因60阶段上限停止；不派发到控制，
不授予物理或任务资格。下一步核查末端旋转、活动约束与响应失配；完整指标、
成本与来源见[cap-distance报告](../reports/tracking/trajectory_search/cap_distance_gn_20261102.md)。

保存模型诊断显示安全约束与局部梯度显著竞争，不能仅凭接近global盒解释失败。
随后固定单旋转方向诊断完成1QP、两侧6响应并通过独立数值/源码审核：旋转误差
.13113→.12820rad，但制动安全slack为负，整体cap loss变差，候选未下发。
75项专项通过。下一步区分Js局部失配与有限步曲率，原安全门保持；见
[约束诊断](../reports/tracking/trajectory_search/cap_distance_constraint_diagnostic_20261102.md)和
[旋转探针报告](../reports/tracking/trajectory_search/rotation_constraint_probe_20261102.md)。

固定缩步/小扰动诊断已完成：两侧16响应全部独立核验、85项专项通过。两个小
尺度的一侧安全斜率约-8.9e-5，旧模型约-7.2e-6，支持已测方向的局部安全模型
失配；不能只归因于大步曲率。alpha=.125及更小虽数值安全，完整caps仍失败，
未下发候选。下一步验证因果安全模型局部修正及新方向，原门槛保持；见
[局部性报告](../reports/tracking/trajectory_search/rotation_direction_locality_20261102.md)。

一次原有因果安全更新在保存的小尺度样本上改善预测，独立实现逐值一致；但随后
新方向完整复算仍不安全：min slack=-5.50e-6，caps13/24、prefix2、witness0。
实际1QP/1NNLS及生产/独立各3响应（共6/1800内步/1874查询），46项专项通过，
未下发候选。秩一修正不保证其他方向安全，下一步需判别新方向局部失配与大步
误差；见[修正探针报告](../reports/tracking/trajectory_search/corrected_rotation_probe_20261102.md)。

修正后方向固定8样本的独立复算已完成，共16响应/4800内步/4994查询，0QP/拟合。
两小尺度斜率约-7.9e-6/-7.7e-6，模型-7.2e-6，未重现旧方向12倍失配；半步数值
安全，但所有非零尺度完整cap loss更差，完整witness仍0。24项相关QA通过。
下一步回到全部位置/旋转联合目标，不再只围绕尾部旋转下降选方向；见
[修正后局部性报告](../reports/tracking/trajectory_search/corrected_rotation_locality_20261102.md)。

恢复完整cap-distance目标的一次QP与固定五尺度验证完成（1QP/1NNLS、14响应/
4200内步/4370查询，47项专项通过）。alpha .25及更小尺度数值安全且联合loss
下降，但caps仍12/24、witness0；安全余量消耗会再次触发原restore模式。
小步预测误差仍明显，下一步核查当前anchor完整局部模型重建的采样域与数值条件，
不靠反复单方向修正代替；见[完整目标探针报告](../reports/tracking/trajectory_search/corrected_cap_distance_probe_20261102.md)。

anchor85的±1e-4单轴探针只能通过64/72轴原command门；固定连续区间联动枚举
可获得72维、condition约4.049且各因果前缀满秩的采样域。几何选择/因果拟合
基础实现及24项专项通过，并逐值复现实际anchor的72方向；尚无新机器人响应、
实际局部模型重建或控制资格。下一步先登记并行采集/独立审核预算再执行；见
[采样域报告](../reports/tracking/trajectory_search/anchor85_sampling_domain_20261102.md)。

anchor85的72联动探针已实际采集并独立复算：148响应/44400内步/46209查询、
33model，32不同worker均exit0且回收；31项专项通过。Js/Jq/Je新模型及独立正规
方程重建通过原1e-10审核，完整响应/anchor/repeat逐值一致。13个非线性不安全
数值探针如实保留，未下发。已见联合alpha .25方向安全预测误差由3.80e-5降至
8.50e-8，但该后验比较不授予新方向资格。下一步用新模型生成完整目标方向并另行
验证；见[联动局部模型报告](../reports/tracking/trajectory_search/anchor85_coupled_stencil_20261102.md)。

新模型生成的联合方向已完整审核（1QP/1NNLS、14响应/4200内步/4370查询、43项
专项通过）。alpha .125下降2.56e-4、实际/预测比.991、reserve .929，保持原pose
进入条件；但旧顺序会先接纳alpha .5并重新restore。全部caps仍12/24、witness0。
下一步显式设计保护reserve的接纳变体及有限迭代比较，不追认旧规则已选较小步；
见[刷新模型新方向报告](../reports/tracking/trajectory_search/refreshed_cap_distance_probe_20261102.md)。

显式reserve接纳与原根域继承已实现，默认55QP/122响应软件回放保持完整答案
SHA不变；新方向单stage回放按原顺序选alpha .125、reserve .929，独立审核及
三类篡改拒绝通过。29项专项通过，尚无新响应/新有限搜索，不能将软件回放计为
控制进展或连续收敛证据；下一步独立登记有限迭代及完整响应审核，见
[接纳变体实现报告](../reports/tracking/trajectory_search/reserve_entry_acceptance_implementation_20261102.md)。

reserve接纳的实际匹配有限对照已完成且两侧独立审核通过：baseline60stage/91生产
响应、reserve7stage/28生产响应，共238双侧响应/71400内步/74260查询、4model、
67QP/67NNLS。baseline loss2.007688，reserve2.009812；两组caps12/24、witness0。
reserve无restore但只接纳1步，后续安全reserve拒绝未满足原反馈资格，缩步至下限。
因此不采用为优胜默认；下一步核查这些已付费拒绝数据的模型校正作用。40项最终
专项通过，所有进程回收；见[实际有限对照报告](../reports/tracking/trajectory_search/reserve_entry_search_20261102.md)。

后验检查不支持盲目增加reserve拒绝反馈：最小安全拒绝样本原Js误差仅2.81e-8，
秩一更新未稳定改善其他尺度。更直接的是原pose QP只保留半数reserve，而实际
接纳要求.9；24个reserve拒绝的模型预测均已低于该门。停滞anchor .928585时，
最小alpha .0625预测.899566，半径缩小仍贴原下限。下一步显式对齐QP与接纳条件，
保留旧默认和原门；见[后验核查](../reports/tracking/trajectory_search/reserve_rejection_shadow_20261102.md)
与[下限对齐设计](../plans/v0.0.1/tracking/trajectory_search/reserve_aligned_floor_design_20261102.md)。

pose下限对齐策略已实现并完整验证一次新方向：1QP/1NNLS、14响应/4200内步/
4370查询，43项最终专项通过，旧默认55QP/122响应回放完整答案SHA保持。
首个原规则合格步alpha .5下降1.88e-4、reserve1.056（旧对应步.855），无需立即
restore；但全部caps仍12/24、witness0。下一步另登记有限连续迭代验证，不能用
单步效果宣称解决停滞；见[下限对齐新方向报告](../reports/tracking/trajectory_search/aligned_cap_distance_probe_20261102.md)。

下限对齐的实际有限迭代已独立审核通过：9stage/4接纳/0restore，最终loss2.009187、
reserve .911682，仍caps12/24、witness0，radius到下限。新费用44双侧响应/
13200内步/13730查询、2model、9QP/9NNLS；50项专项通过，旧baseline60QP/91响应
闭合回放完整答案不变。相比只加接纳门有更多下降，但未解决停滞；末段预测reserve
.9088而实际.8901，下一步核查当前安全拒绝的临界模型误差/校正，不能复用旧结构
不一致解释。见[连续迭代报告](../reports/tracking/trajectory_search/aligned_reserve_search_20261102.md)。

对齐后stage5的最小安全reserve拒绝response13用于后验更新时，独立实现逐值一致；
同方向未拟合alpha .5的Js最大误差由3.90e-6降至2.50e-8，其他已见方向也有不同
程度改善，但仍有reserve假阳性，Jq在最远方向变差。0新响应/查询，未改旧接纳。
下一步仅隔离Js安全反馈，保留Jq/Je及完整实际门，不把后验改善提升为新方向资格；
见[后验校正报告](../reports/tracking/trajectory_search/aligned_rejection_shadow_20261102.md)
与[安全反馈设计](../plans/v0.0.1/tracking/trajectory_search/reserve_safety_feedback_design_20261102.md)。

仅Js的reserve安全反馈已显式实现并独立审核，原pose ceiling来源保持优先、
Jq/Je及anchor不变。最终52项专项通过；六stage保存前缀回放选择response13并
通过五类篡改拒绝，旧baseline60QP/91响应完整答案SHA保持。0新响应/查询，尚未
验证反馈后的新方向或连续迭代；下一步另登记有限新执行，见
[安全反馈实现报告](../reports/tracking/trajectory_search/reserve_safety_feedback_implementation_20261102.md)。

仅Js反馈实际新搜索已完整native审核：12stage/6接纳/2安全only更新、0restore，
loss2.008818、reserve .903607，仍caps12/24、witness0。新费用40双侧响应/
12000内步/12482查询、2model、12QP/12NNLS；71项专项通过，旧aligned9QP/22响应
闭合回放答案不变。末3stage共15候选未执行，因全步预测下降低于原1e-4而缩步
停止，不能据此证明物理无可行下降。下一步固定核查未调用方向及其完整跟踪意义，
不直接放宽门或仅按微小loss选优；见[实际搜索报告](../reports/tracking/trajectory_search/reserve_safety_search_20261102.md)。

停止前3个固定未调用方向已新增完整核查：10双侧响应/3000内步/3122查询、2model，
0新QP/拟合，43项专项通过。三方向均joint/buffer安全且loss下降，前两reserve
失败；最小方向reserve .901292、下降3.10e-5，低于旧进展门，仍caps12/24、witness0。
这证明有小幅局部下降，不证明累计跟踪改善，也未追认旧接纳。下一步仅作为显式
新算法对照设计更小进展门、保持原实际安全和完整资格及最大预算；见
[固定方向报告](../reports/tracking/trajectory_search/deferred_cap_directions_20261102.md)与
[进展门对照设计](../plans/v0.0.1/tracking/trajectory_search/small_progress_comparison_design_20261102.md)。

迁移前实现阶段的专项、样式及真实输入记录见
[实现准入报告](../reports/tracking/trajectory_search/state_vector_dual_recovery_implementation_20261029.md)。
最终源码的Ruff/format、结构、导入图、179项修复复核、兼容环境40项和两真实
dry通过；5177项发现及顺序保持。全量v2、后续修复和最终专项分别登记，原5项
PyArrow/NumPy导入错误与4项已复现数值失败不改容差、不伪装通过。最终验收后
当时仅更新本状态、迁移说明和PR文案，未再修改源码、测试或原始输入；上述
2026-11-02 研究实现与专项检查单列，不继承迁移阶段全量验证结论。

2026-11-11：以独立30条训练demo的D0缓存拟合单一累计位姿/开度参考校正；在六条
已查看validation demo的48条已保存raw响应上，原054离线联合指标门通过，H8/H16
联合分数改善11.31%/13.81%，位置小幅退化仍在原保护内。独立数值重算通过；没有
新WAM、物理、GPU或fresh，尚未证明机器人跟踪、抓取或任务成功。见
[校准报告](../reports/wam/training/cached_endpoint_calibration_20261111.md)。

同日对18条正确参考和6条raw开发轨迹完成固定历史影子检查：校正使正确参考
相对旧实测轨迹的误差增大，6条开发轨迹前8点没有新增低开度chunk，其中task1
seed43的75个完整窗口仍无低开度预测。这不是新物理或任务结果，见
[影子报告](../reports/wam/training/cached_endpoint_shadow_20261111.md)。

同日三任务6分支实际夹爪短窗已完成并独立重放：校正组在task0相对原WAM参考仅
3/8端点达位姿门，task1派发0步，task2的2步中1步达标；20个新周期无grasp或任务
成功。原checker曾在物理前因策略导入隔离失败，独立修复重放通过，原失败保留。
该候选未获统一控制资格，见[物理反馈报告](../reports/tracking/joint_control/cached_calibration_gripper_feedback_20261111.md)。

另以30条训练demo做五折因果夹爪事件筛查：八个预设阈值均未同时满足误闭合、
开度误差和跨任务命中条件，未拟合最终候选或读取validation未来标签；见
[事件训练报告](../reports/wam/training/causal_opening_event_20261111.md)。

同一已生成参考中后16点存在闭合时才允许提前的探索，在原五折训练筛查中按预定
规则选择阈值0.7；H8新增11个真实命中和11个误闭合，task2无新增命中。一次最终
拟合后的48条保存validation响应未通过开度误差及跨任务门，独立复算确认失败；
监督器因结果文件名错误退出，拟合和评分子进程均exit0，原owner退出收据缺失。
未执行新物理，见
[支持事件筛查](../reports/wam/training/supported_opening_event_20261111.md)。

另对原策略的因果暂缓闭合提案完成历史影子：18条正确参考4907周期触发1次，
12条开发轨迹7200周期触发10次，六条raw无触发；已研究task1短窗另触发1次。
正确参考触发处原基线已有物体grasp，必须先做受载物理对照；影子不能证明候选
抓取或跟踪，见[暂缓闭合影子报告](../reports/tracking/gripper/defer_close_release_shadow_20261111.md)。

correct_08受载点成对CPU物理对照及独立重放已完成：step85唯一夹爪命令由+1改0，
后三端点平均绝对opening误差0.006419→0.005026，两组8端点均有grasp、无任务
成功；最大TCP位置/旋转误差小幅增加。反事实后两条命令固定为历史动作，不是
在线反馈或18条正控资格，见[受载点报告](../reports/tracking/gripper/defer_close_correct_feedback_20261111.md)。

已研究task1释放短窗的实际反馈及独立重放已完成：原WAM opening RMSE由0.280611
降至0.204035，最后误差仍为−0.296015，两组均无目标grasp或任务成功。该窗口是
假设来源而非独立新初态；见[反馈报告](../reports/tracking/gripper/defer_close_task1_feedback_20261111.md)。

同日 correct_08 两个连续参考窗口的统一反馈及独立物理重放已完成：16/16 个端点
达到所登记位置和旋转门，任务成功仅最后 1 步，不构成稳定成功；见
[连续反馈报告](../reports/tracking/joint_control/ordered_correct_stream_feedback_20261111.md)。
独立重放后已捕获 step80/88 的当前双相机图像，保留首次路径错误和重放预算；
见[观测报告](../reports/tracking/references/ordered_feedback_wam_capture_20261111.md)。
两次 WAM 请求已产生响应，但原工作者在结果汇总时因单请求掩码检查错误 exit1，
逐请求初始噪声 ledger 未保留；独立 CPU 旁证仅报告参考差异、不改写原失败，见
[模型旁证](../reports/tracking/references/ordered_feedback_wam_partial_audit_20261111.md)。
同前缀 step80 已保存 WAM 参考的八周期实际反馈及独立重放完成：实际状态相对
WAM 参考的位置 RMSE 为 0.002405 m，相对正确参考为 0.013118 m；前 80 周期是
正确参考历史且原模型收尾失败，不构成完整 WAM 闭环。见
[短窗报告](../reports/tracking/joint_control/saved_wam_feedback_20261111.md)。
已按该 WAM 短窗实际动作重放并捕获 step88 当前图像，保留首次回调失败及累计
预算；该图像与正确参考轨迹的 step88 不同。见
[当前观测报告](../reports/tracking/references/wam_feedback_current_observation_20261111.md)。
随后在同 step80 图像上重新生成并逐元素匹配已实际执行的参考，才以 WAM 反馈
到达状态的 step88 图像生成第二参考；完整模型及独立审计已通过，首次执行器
上下文失败保留。见
[重观测报告](../reports/tracking/references/wam_feedback_reobserve_20261111.md)。
随后已执行并独立重放两段因果绑定参考的 16 周期反馈，两个窗口均 8/8 满足相对
WAM 位姿门，末尾连续 9 个端点满足任务谓词，尚未达到 40 端点稳定门；前 80 周期
仍为正确参考历史。见[两窗口报告](../reports/tracking/joint_control/causal_wam_two_window_feedback_20261111.md)。
