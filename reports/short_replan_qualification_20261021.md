# 8/16 步重查询的 oracle 执行资格 — 2026-10-21

状态：固定十二条 CPU 资格正在执行，最终独立门禁尚未生成。本报告不计入任何 WAM 成功率；不存在新增模型请求或新初态评测。

[007 计划](../plans/v0.0.1/007_short_replan_qualification_20261021.md)在启动前冻结十二条/7200 物理步：maxobs8/16 × position_hold无gov/recovering_position_hold × bowl/wine/cream各init0，seed42，每条600步、20Hz、H40/执行40、末40稳定成功、canonical TCP RMSE≤10mm、SO(3) RMSE≤.05rad。原004/005/006执行器、核验器与配置均保持字节不变，失败不重跑或放宽门槛。

新增[编排入口](../scripts/qualify_short_replan.py)与[独立注册核验入口](../scripts/verify_short_replan.py)在 `841abfc` 提交。四组实际matrix入口dry-run均返回三条oracle任务、model_started=false、physics_started=false；两个新脚本的Ruff及格式检查通过，未注册配置被拒绝。

准备阶段冻结248项输入，保存28份源码/配置副本。[launch manifest](../outputs/wam_stability_20261021/short_replan_qualification/qualification_launch_manifest.json) SHA为 `8fb2376ec303bafc5f40864d7734769bbbc92ee51dd9cc57a400dbf493e578e2`。源码、配置、正确参考及外部实际模拟器源均在执行前后核验，编排四组依次运行，每组最多两个CPU/OSMesa worker；CUDA_VISIBLE_DEVICES及LD_LIBRARY_PATH为空，不启动模型。

启动新物理前，使用已提交的 `audit_wam_position.verify_case(full=True)` 只读重放004的两种v3×三个场景共六条旧maxobs80结果，全部通过。该路径同时重算独立scalar GripperReplay、RecoveryReplay、原始/有效/准入/nominal参考、OSC、原生motor积分和0保持、qpos开度、查询时钟和参考来源。[prior80 gate](../outputs/wam_stability_20261021/short_replan_qualification/prior80_independent_verification.json) SHA为 `1ba534198a8eb42cfac6b505c7dd72d29fcd5fb1216f3425bd511768b358bd30`；这六条是原始结果回放，没有新增物理步。

后续完整门禁将逐条独立重算7200端点并与同variant、同场景的旧80步参照比较query数、bridge比例、实际模型游标消费和跟踪误差。正确参考按当前物理step索引已有native世界端点，属于特权oracle资格；即使全部通过，也不证明短间隔WAM能消费有效关闭参考或稳定完成任务。
