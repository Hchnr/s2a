# 末周期平移探针：在原安全门槛内恢复终端余量

本文件为下一阶段执行前登记，尚未实现或执行。本轮只固定有限采样、来源、选择及资格规则；前项保存分析及 E17 已闭合绑定，仍须完成新实现、独立检查和 CPU QA 才可执行。本阶段不运行 QP、物理 pilot 或 GPU。

## 已闭合前项与固定来源

前项[终端单行割线外推](terminal_secant_probe_20261029.md)的17次完整响应及独立复算已经关闭。以下路径均相对仓库根目录，执行前须绑定实际文件字节、各 owner 的实际 wait/reap、QA 与来源闭包；科学资格失败不等于执行记录不完整。

| 来源 | 路径 | SHA256 |
| --- | --- | --- |
| producer | `outputs/tracking_v5_terminal_secant_probe_20261029/verified.json` | `39aff2ebda4d272b1b03e0cfe60e5bf26a69de10c4870f55afa4280052b98c34` |
| producer owner | `outputs/tracking_v5_terminal_secant_probe_execution_20261029/result.json` | `ffd7763b90f18fd16cbde76bcce5827e102fb948bd118d4439b4788d44ffb380` |
| 独立 checker | `outputs/tracking_v5_terminal_secant_probe_check_20261029/verified.json` | `06214022d72f13423c8829344630ed632e4f9a1b6a40125e0aeb2a2e222464ba` |
| checker owner | `outputs/tracking_v5_terminal_secant_probe_check_execution_20261029/result.json` | `cdf6e32104efc5f8344c51c5176eee87a3ebe3a80444dcfdff2938c88d1305a3` |
| 前项 QA | `outputs/terminal_secant_probe_native_qa_20261029_2wj0oaca/result.json` | `28db556f5af66b66ad915cd2575ed89cd8e830c5d830b9b6564e534cc1c39019` |
| 完整闭合 | `outputs/terminal_secant_probe_final_closure_20261029.json` | `29e3b90d5f0919044abba53102411816f6b78a648a3509a5602a58f64436d779` |
| 全15点保存分析 | `outputs/terminal_secant_probe_analysis_20261029/result.json` | `44d9c06da3824ab17755750ac195ce60152d15e79fd536d17492d9b33a1268e0` |

完整闭合记录为2422来源，checker 完成17响应且 strict 分类差异为空。全15点分析正常关闭，两个混合方向的原定外推资格均失败；这些点虽命令准入通过，纯 v 与 e+v 各5点均仅 prefix5，e-v 的5点均仅 prefix4，完整 witness0。新阶段从该 producer 的 `answer.geometry.anchor_commands` 取得固定 H12 命令；该命令仍绑定旧 SQP accepted anchor1819，不能改用其他 selected 命令。历史完整响应取同 producer 的首 anchor。E17 必须由上述前项全15个新点的原模型残差计算并与保存分析交叉核验，禁止使用本阶段12个新点拟合或调整阈值。

## 问题与保持项

检验仅改变最后第12周期的3个平移命令，能否在全部原安全约束和已达标 prefix9 保持的条件下，提高 `terminal[0]`。原 terminal normal 为 e7，故该值为 `-final_joint_velocity[6]`，提高它意味着降低关节7的末速度。旧普通 SQP 接纳还要求原失败 cursor 的误差下降，纯终端恢复不满足该接纳条件，见 [accepted_point_sqp_plan.py:434](../../src/s2a/accepted_point_sqp_plan.py#L434)。本阶段单独记录恢复，不伪装成该算法已经接纳的推进步。

保持原 H12/600ms、完整12参考点、TCP/坐标系、位置 `.01m` 与旋转 `.05rad` caps、原关节范围/制动时间 `.1s`/terminal normals、原全部审计容差及 `1e-8` 优化安全 buffer。固定完整 XML/资产快照、2ms/25内步控制器、原生 float32 旋转目标算法、M/J 更新、力矩裁剪、增益及被动力规则；不使用割线修正后的模型。保持 actual368 的 q/v、fixed finger、initial_joint、previous completed367 command 与 completed367 latch，以及原已生成参考、夹爪计划和当时可用的历史误差 profile；不使用 actual369+反馈或新增历史样本。

每次调用都从同一 actual368 初态重新复制并完整响应12周期，不能把 anchor 的预测终态当新物理初态。前11周期命令及末周期3个旋转命令保持原字节；这使前11周期响应与末周期初态应完全相同，且所有12个旋转 goal 应完全相同，见 [native_goal_trajectory.py:52](../../src/s2a/native_goal_trajectory.py#L52) 和 [native_goal_joint_response.py:139](../../src/s2a/native_goal_joint_response.py#L139)。末周期平移仍可能改变后续力矩裁剪及状态相关动力学，不能据此归因旧误差唯一来自旋转量化。

## 固定14次请求与模型前准入

按以下顺序执行：首 anchor；轴 x、y、z 依次各执行 `+0.001`、`-0.001`、`+0.0005`、`-0.0005`；强制末 anchor repeat。步长单位为 normalized command，每条只改 `commands[11, axis]` 一个 float64 元素。保存 nominal displacement、实际浮点 displacement、命令 bytes 与请求索引。12候选互异、均与 anchor 不同，末 repeat 与首 anchor bytes 相同；共14计费调用、13唯一命令，无缓存跳过或结果复用。

模型构造/当前查询前，一次性独立检查 anchor 和全部12候选的有限性、形状、仅指定元素变化、精确生成顺序及原 box/rate 准入。rate 为相邻命令各轴变化不超过 `.25`，首周期相对 completed367；原容差 `1e-12` 保持。任何候选不准入则在0响应/0查询时停止并保存原因；不裁剪、不跳过非法点继续、不改符号/步长、不追加试验。实际位移为零或唯一性失败同样拒绝。

当前只读元数据观察到末周期平移的有效负/正余量如下，计算为原 box 与上一周期 ±.25 的交集；它们只是该固定输入的观察值，不硬编码为验收常数。12候选当前均有命令准入余量，响应安全尚未测得。

| 轴 | 负方向余量 | 正方向余量 |
| --- | ---: | ---: |
| x | 0.4704189141965543 | 0.029581085803445706 |
| y | 0.3984407581476924 | 0.10155924185230755 |
| z | 0.07281734038686194 | 0.20727249008356674 |

通过准入后完整执行全部14次，不因安全失败、余量达标或任何 witness 标签提前停止。每次300数值内步、312查询，另阶段首次当前查询1；producer 固定4200内步/4369查询，独立 checker 也逐次完整复算14条（含 repeat），另收费4200/4369。两阶段共28次完整数值响应、8400内步、8738查询；优化迭代0。技术异常保留实际已完成计数及失败记录，不自动重试；科学检查不通过仍保留全部已完成数据。

## 完整响应、因果检查与选择

保存全部14条完整 rollout、每周期 goal/torque/clipped、完整九组原审计、五组非线性原始 safety_slack、12点 TCP 误差与归一化平方误差、原 strict joint/prefix/full-witness 标签。不得将 `restoration_eligible` 覆盖原 `feasible`：后者仍表示完整H12原caps及原审计全部通过。

逐条与本次首 anchor 核对前11个完整 period 响应，包括 q/v、FK endpoint/twist、目标、M/J变化摘要及请求/实际力矩等所有原已保存数组；同时核对顶层前275内步和前11端点。所有12个旋转 goal 必须 exact 相同。新首 anchor 与前项完整历史 anchor、首末 anchor 也分别进行完整 canonical JSON 数值数组逐路径 exact/maxdiff 比较；不声称 JSON 保存了原 dtype 或空数组隐藏维度。任何所需 exact 检查失败均保留差异，恢复资格为 false，不能靠放宽容差、平均重复或再运行消除。跨 producer/独立 checker 的数值比较仍沿用既有原容差，strict 分类不放宽；每条执行内部的因果 exact 要求独立检查。

候选 `restoration_eligible` 要求：原完整关节/命令审计通过；`inner_lower/inner_upper/braking_lower/braking_upper/terminal` 全部原始 slack 均 `>=1e-8`；原 prefix9 的9点各自位置/旋转归一化平方误差均 `<=1`；所需历史、repeat、因果 exact 检查全部通过。所有12点误差照常报告，不把尚未达标的后三点掩去。原审计内部容差保持，但不能据其容差替代额外 `1e-8` buffer。

在12个 `restoration_eligible` 候选中按实际 `terminal[0]` 降序选择，完全相同时依次按较小实际扰动 L∞、较小固定请求索引决胜。不用近似相等处理排序。若无合格候选，或最佳值不严格大于本次首 anchor，则 `selected=None`、`improvement_found=False`；保留候选排名与拒绝原因。否则记录被选完整响应、终端值/余量/改善、其他约束最小值和所有 pose 误差；选择不依赖下述足量阈值，也不在不同 h 之间另拟合或追加搜索。

## 足量恢复阈值与结论边界

定义前项15点的 `E17=max(abs(original_residual_i))`，其中 original_residual 使用旧未修正 terminal 行、各点实际命令位移，以及前项新首 anchor 为基准的实际 terminal 变化；包含纯 v 与两个混合方向、全部5尺度，不排除 clipping 改变或表现差的点。前项已闭合分析给出固定 `E17=2.3626357467448553e-6`，因此 `reserve_target=10*E17`，对应终端 slack 门槛 `1e-8+10*E17=2.3636357467448554e-5`。执行前将前项分析 ref、15值重算结果及该固定阈值一并绑定到登记；不能仅信任手写数值而不核验来源。

`sufficient_reserve_restored` 仅在 `selected` 存在且 `terminal_selected - 1e-8 >= 10*E17` 时为 true；全部原安全、因果及 repeat 条件仍须通过。该10倍系数和取全15最大值的规则在新响应前固定，不根据新结果选择、更换为修正模型残差或降低门槛。低于该阈值的正改善如实记为可行改善，不能冒充足量恢复。

`10*E17` 只是依据已观察有限尺度预测误差设置的经验预留目标，不是72维未采样命令误差上界、鲁棒安全保证或导数存在性证明。新方向不同，满足该门槛也不证明下一次 QP 可接纳。前11周期完全相同意味着 cursor9 的误差完全不变；本阶段只能检验终端恢复空间，不能计作参考推进、完整轨迹解或任务成功。12个点都失败也不能证明整个72维可行域无解。

## 独立核验与后续边界

新独立 checker 不导入生产候选生成/选择 helper，独立重建12个命令、计账、全部原审计及阈值/排序；使用已核验独立原生数值闭包从相同因果初态完整复算14次，检查所有保存字段、全部 strict 标签、历史/repeat/因果 exact 与 E17 前项绑定。恢复科学标签可为 false，而完整性检查仍应如实报告通过；源或数值不一致不能被科学失败掩盖。实施时新增来源及必要合成 QA 应冻结并经两入口实际 dry，使用独占仓库内输出目录和实际 wait/reap 记录，不覆盖旧产物。

本阶段新增物理、world-model、fresh、参考消费、实际推进及优化迭代均0，`control_qualified=False`；actual cursor仍8，task/grasp仍0。命令从本仓库执行，CUDA/LD显式空、原单线程 CPU 环境及 seed42 保持，不写旧清理根或外部共享路径。原18正确参考/12WAM开发/24最终留出门槛不变。后续即使找到足量内点，也须另行登记从该点继续规划的预算与资格；本计划不直接启动 QP、改变固定控制器或进入物理 pilot。
