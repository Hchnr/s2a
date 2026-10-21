# 8/16 步重查询的 oracle 执行资格 — 2026-10-21

固定十二条 CPU 资格全部完成并通过独立门禁：12/12末40端点稳定成功，7200个物理步，无执行错误。maxobs8/16各六条均满足预注册的TCP/SO(3)门槛。本报告不计入任何WAM成功率；新增模型请求和新初态评测均为0。

[007 计划](../plans/v0.0.1/007_short_replan_qualification_20261021.md)在启动前冻结十二条/7200 物理步：maxobs8/16 × position_hold无gov/recovering_position_hold × bowl/wine/cream各init0，seed42，每条600步、20Hz、H40/执行40、末40稳定成功、canonical TCP RMSE≤10mm、SO(3) RMSE≤.05rad。原004/005/006执行器、核验器与配置均保持字节不变，失败不重跑或放宽门槛。

新增[编排入口](../scripts/qualify_short_replan.py)与[独立注册核验入口](../scripts/verify_short_replan.py)在 `841abfc` 提交。四组实际matrix入口dry-run均返回三条oracle任务、model_started=false、physics_started=false；两个新脚本的Ruff及格式检查通过，未注册配置被拒绝。

准备阶段冻结248项输入，保存28份源码/配置副本。[launch manifest](../outputs/wam_stability_20261021/short_replan_qualification/qualification_launch_manifest.json) SHA为 `8fb2376ec303bafc5f40864d7734769bbbc92ee51dd9cc57a400dbf493e578e2`。源码、配置、正确参考及外部实际模拟器源均在执行前后核验，编排四组依次运行，每组最多两个CPU/OSMesa worker；CUDA_VISIBLE_DEVICES及LD_LIBRARY_PATH为空，不启动模型。

启动新物理前，使用已提交的 `audit_wam_position.verify_case(full=True)` 只读重放004的两种v3×三个场景共六条旧maxobs80结果，全部通过。该路径同时重算独立scalar GripperReplay、RecoveryReplay、原始/有效/准入/nominal参考、OSC、原生motor积分和0保持、qpos开度、查询时钟和参考来源。[prior80 gate](../outputs/wam_stability_20261021/short_replan_qualification/prior80_independent_verification.json) SHA为 `1ba534198a8eb42cfac6b505c7dd72d29fcd5fb1216f3425bd511768b358bd30`；这六条是原始结果回放，没有新增物理步。

完整门禁逐条独立重算7200端点，重新核验351项输入SHA，全部通过；四组coordinator退出码均0。没有修改候选参数、运行中源码、旧结果或重跑失败。正确参考按当前物理step索引已有native世界端点，属于特权oracle资格；末端不足40个有效未来时只给真实前缀，不用padding假冒未来。该oracle可在重query时重新连接真实世界端点，不能当作WAM能选择正确未来的证据。

| 间隔 | 方式 | 场景 | 稳定 | TCP RMSE mm | SO(3) rad | query数 | bridge端点/600 | 累计消费模型秒 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 无gov | bowl | 是 | 1.838 | .003339 | 75 | 0 | 30.000 |
| 8 | 无gov | wine | 是 | 1.421 | .006744 | 75 | 0 | 30.000 |
| 8 | 无gov | cream | 是 | 1.548 | .002595 | 75 | 0 | 30.000 |
| 8 | recovering | bowl | 是 | 2.459 | .003752 | 75 | 0 | 26.994 |
| 8 | recovering | wine | 是 | 1.916 | .007153 | 75 | 0 | 27.825 |
| 8 | recovering | cream | 是 | 2.271 | .003692 | 75 | 2 | 27.295 |
| 16 | 无gov | bowl | 是 | 1.838 | .003339 | 38 | 0 | 30.000 |
| 16 | 无gov | wine | 是 | 1.421 | .006744 | 38 | 0 | 30.000 |
| 16 | 无gov | cream | 是 | 1.548 | .002595 | 38 | 0 | 30.000 |
| 16 | recovering | bowl | 是 | 2.163 | .005279 | 38 | 1 | 28.127 |
| 16 | recovering | wine | 是 | 1.339 | .006839 | 38 | 0 | 28.991 |
| 16 | recovering | cream | 是 | 1.833 | .003804 | 38 | 0 | 28.537 |

TCP实际范围1.339–2.459mm，SO(3) .002595–.007153rad。所有query均消费了正的模型时间；本组正确参考没有出现反复重query而整query保持模型时间0的情况。累计模型秒是各chunk独立游标消费量之和，不是示范绝对时刻或物理任务进度。

旧004 maxobs80三场景无gov与recovering均各15query；无gov累计模型30秒，recovering分别28.700/29.100/28.800秒。新8步recovering消费略降，16步接近旧值，同时均保留稳定任务成功。bridge采样端点占比最高2/600=.333%；某些桥在一个控制周期内完成，bridge flag为0不等于没有桥。由保存pose独立计算的桥段retimed总时长，8步三场景为.276/.330/.339秒，16步为.197/.131/.145秒；这不是连续物理时间测量，不能把它与所有时钟损耗简单相加归因。

无gov的oracle每个物理step都指向同一世界参考，原则上重query频率不应改变逐步控制。完整比较两新间隔×三场景的600端点与旧80步结果，最大绝对差：desired TCP矩阵`5.11e-15`、measured/after source矩阵`3.40e-12`、command `9.06e-11`、opening `5.87e-13`、qpos `4.64e-14m`、对象位置`1.65e-13m`。motor target/opening差为0，夹爪command/state/reason/normal_intent/closing_episode及逐步success历史完全相同。这是数值舍入量级的复现，未加入新门槛，也未因此重跑。每trial的同一adapter跨chunk保留历史，previous command和motor连续关系已独立核验；不能把无gov的成功重复视为额外任务能力收益。

十二条在首次新重query之前的首8/16端点全部保存字段、首次完整生成oracle chunk、初始controller和XML也与旧对应variant精确一致。四个coordinator、十二个physics worker与一个编排器的17个记录PID均已退出；采样进程快照显示空CUDA/LD、OSMesa与两个worker。额外只读补证再次绑定369项输入，未追加物理或模型执行。

| 完整证据 | SHA256 |
| --- | --- |
| [最终独立门禁](../outputs/wam_stability_20261021/short_replan_qualification/qualification_independent_verification.json) | `64f0eda45d3e5495dc12124795047beade66c6a3489c6b1f8061a637af9dc4ad` |
| [执行启动身份](../outputs/wam_stability_20261021/short_replan_qualification/qualification_execution_launch.json) | `1fe6859870f37ed5bd8cd8e8e3d63268fe7b62200eaf08abc4041bbdb9fdbac4` |
| [首段/桥时长/进程退出补证](../outputs/wam_stability_20261021/short_replan_qualification/physical_prefix_and_exit_verification.json) | `c24459dc4aa232672266ac1f9aeba7e81e8a8c70b8efda0cffd0f3d2f97e87d2` |
| [无gov完整600端点对照](../outputs/wam_stability_20261021/short_replan_qualification/nogovernor_world_reference_comparison.json) | `60b2e22d848bb833eab944a4ecbd0f0638fb995fabbcd330318896dfd7b870d1` |

两种短间隔及两种v3执行方式通过了这些旧场景上的正确参考执行资格，没有认证任意WAM轨迹、握持力、机器人泛化或真机。新WAM矩阵的checkpoint、请求/物理预算和方案仍需另行冻结；本资格不会自动启动模型。短间隔WAM中仍可能有错误几何、参考后缀丢弃或桥比例增加，须由独立闭环结果回答。
