# 最终验收差距只读审查 — 2026-10-23

当前证据尚未证明总目标。最终口径保持为原预留12个新初态×sampling seed42/43的24条真实WAM闭环，每条600个20Hz端点，最后40端点连续满足原生任务谓词；至少20/24稳定、每任务至少4/8。[036:7](../plans/v0.0.1/036_project_goal_20261023.md#L7)、[036:25](../plans/v0.0.1/036_project_goal_20261023.md#L25)

本次只读取现有metadata、源码与小文件指纹，生成[机器收据](../outputs/wam_final_acceptance_gap_audit_20261023/review.json)。未启动模型、物理、训练或fresh任务，未解码NPY/NPZ/HDF5未来数组，未更改冻结实现/配置/计划，未根据最终数据选择参数。收据的完成表示审查完成，不表示候选或最终验收通过。训练运行的进程生死、退出原因及完整资格由所属owner另行核实，本审查不以结果文件缺失推断进程已停止。

| 最终要求 | 当前依据 | 尚缺的直接证据 |
| --- | --- | --- |
| 原12个新初态，每任务4个；每初态42/43 | [预留配置:16](../configs/wam_stability_fresh_reservation_20261021.json#L16)；fixture仍为`6e279c4d…134c6`；本次重核12个initial SHA唯一、与5份历史fixture的18个初态无交集，12份expert文件仅按字节核SHA | 同一12×2矩阵实际24条完整运行；metadata预留不能计成功率 |
| 同一候选覆盖全部任务/seed | [036:25](../plans/v0.0.1/036_project_goal_20261023.md#L25)；038规定W2为唯一候选、开发稳定门后才能另行冻结fresh协议，[038:112](../plans/v0.0.1/038_clean_residual_training_20261023.md#L112) | 实际W2 fit及独立重算、数值/来源资格、开发任务结果；最终完整候选指纹、控制/采样配置与独立准入。当前040明确`fresh_performance_authorized=false`，[040 config:24](../configs/wam_clean_training_execution_20261023.json#L24) |
| native/oracle/WAM完整矩阵 | [036:30](../plans/v0.0.1/036_project_goal_20261023.md#L30)要求相同初态/控制合同正控；旧native执行器校expert SHA、结束后零arm保持gripper并保存实际after-state参考，[checkpoint:99](../src/s2a/checkpoint_simulation.py#L99)、[checkpoint:139](../src/s2a/checkpoint_simulation.py#L139)、[checkpoint:224](../src/s2a/checkpoint_simulation.py#L224) | 最终72条/43200步的完整冻结矩阵与收据：24 native、24 oracle、24 WAM；预算算术见[准备审查:38](wam_fresh_acceptance_preparation_review_20261021.md#L38)，本次未授权或运行此预算。正控失败保留原24分母，不替换初态或删除病例 |
| 首请求确为该新初态的当前物理输入 | 位置执行器保存actual initial physics SHA、step0当前measured anchor及推理前后state/time，[position:229](../src/s2a/stability_position_simulation.py#L229)、[position:243](../src/s2a/stability_position_simulation.py#L243)、[position:279](../src/s2a/stability_position_simulation.py#L279)；014检查step0与fixture初态一致，[014:240](../scripts/wam_position_acceptance_checker.py#L240) | 每个initial/seed自己的raw双相机RGB witness、渲染参数、无推进物理时间证明、独立canvas重建及fresh callback；旧缓存PNG或003/006预期raw不能冒作fresh首输入。合同见[准备审查:28](wam_fresh_acceptance_preparation_review_20261021.md#L28) |
| 600步、20Hz、物理暂停推理；不因success早停 | 位置执行器循环完整budget、检查控制dt，推理前后物理不变，[position:187](../src/s2a/stability_position_simulation.py#L187)、[position:235](../src/s2a/stability_position_simulation.py#L235)、[position:279](../src/s2a/stability_position_simulation.py#L279) | fresh实际600条端点/每次30s物理时钟；完整请求、响应、命令、source/TCP desired/after-state、开度、SO(3)误差、限幅与实际dt日志；代码路径存在不能代替运行 |
| 最后40端点连续原生success | `object_progress`的success来自`env.check_success()`，[checkpoint:29](../src/s2a/checkpoint_simulation.py#L29)；`task_outcome`要求完整boolean预算并取tail all，[task_acceptance:6](../src/s2a/task_acceptance.py#L6)；014独立重算600/tail40，[014:510](../scripts/wam_position_acceptance_checker.py#L510) | fresh完整600个原生布尔值及独立重算；ever/terminal、object/contact诊断或短暂成功均不能替代 |
| 总体20/24且各任务4/8，按12初态报告相关重复 | [036:29](../plans/v0.0.1/036_project_goal_20261023.md#L29)、[036:33](../plans/v0.0.1/036_project_goal_20261023.md#L33)；聚合工具按task×scene×seed保留scene相关性，[task_acceptance:51](../src/s2a/task_acceptance.py#L51) | 完整24条固定分母分组表、task计数及按初态分组的不确定性；当前不得报0/24或通过率，因为最终性能尚未执行 |
| 独立来源/因果/控制/数值/预算/生命周期审计 | 014逐请求核seed、action domain/mask、响应和反归一化，[014:131](../scripts/wam_position_acceptance_checker.py#L131)；逐步核因果重观测和实际dt，[014:324](../scripts/wam_position_acceptance_checker.py#L324) | fresh专用独立主审计、全矩阵/全部目录核账、PID/birth实际wait/reap/释放、异常与未执行项；已有040/开发审计覆盖各自协议，未覆盖最终24条 |

当前可复用基础与必须处理的接入问题如下。这些是实现任务，不构成最终性能结论。

- Reservation本身只有metadata选择与门槛；明确`candidate_status=Not selected`、未开始资格/评测，[reservation:115](../configs/wam_stability_fresh_reservation_20261021.json#L115)。旧矩阵从其output root直接找fixture，见[run_wam_simulation_matrix.py:33](../scripts/run_wam_simulation_matrix.py#L33)，不能只把reservation作为完整执行配置传入。
- Oracle必须绑定同record、同sampling seed的实际native全600端点。现有`_read_oracle`先找无seed目录、缺失才找带seed目录，[stability_recovery_simulation.py:166](../src/s2a/stability_recovery_simulation.py#L166)；最终新接入须显式核唯一带seed来源并拒歧义，保持既有冻结文件。旧016 wrapper只接受42与旧三record，[audit_wam_tracking_replan.py:40](../scripts/audit_wam_tracking_replan.py#L40)、[其:69](../scripts/audit_wam_tracking_replan.py#L69)，不能直接用作fresh全矩阵审计。
- 首相机witness应沿用实际[render_current_cameras:4](../src/s2a/cameras.py#L4)与[canvas:116](../src/s2a/libero_simulation.py#L116)约定，由独立逻辑重建保存PNG，证明当前输入/响应来源。没有fresh gold输出，来源认证不能声称生成正确，也不需为制造exact额外调用模型。
- 历史fixture保留旧绝对路径和原SHA；本次仅定位新工作区安全副本，并核12份expert字节。最终接入需显式迁移映射并核实际SHA，不向旧目录写入/恢复。[RelocatedArtifacts:21](../src/s2a/relocated_artifacts.py#L21)、[其:50](../src/s2a/relocated_artifacts.py#L50)
- 新038开发矩阵只是旧3初态×42/43×raw/W2的12条，其稳定5/6、各任务1/2是后续候选冻结条件；checker同时保留最终20/24、各4/8并将fresh授权设False，[development checker:87](../scripts/verify_wam_clean_training_development.py#L87)。它没有为最终集合选参或开启性能评测的权限。
- 后续模型入口只可显式使用AGENTS最新允许UUID，CPU/OSMesa空CUDA/LD；014中的旧gpu数值字段仅可作为历史检查兼容标记，不能作为启动映射。[AGENTS:30](../AGENTS.md#L30)、[014:135](../scripts/wam_position_acceptance_checker.py#L135)

本次扫描的预留root只有fixture、12份expert NPY和本地LIBERO配置；没有`result.json`、`steps.jsonl`、query/request/response、首相机witness或服务运行产物。机器收据逐路径保存扫描范围和当前SHA，未把“未发现”扩大为所有外部路径绝无运行的证明。三份大HDF5未重新全量散列、未打开数组；历史完整源资格仍以原metadata审计为准。本次不重新认证预训练未见、历史图像/pose/初态同步或一般可靠性。[metadata边界:17](wam_fresh_reservation_metadata_20261021.md#L17)

因此当前可推进的顺序仍是：恢复对既有D0实际生命周期与已消耗预算的真实记录；按独立资格允许的后续协议完成训练及开发任务；若固定开发门通过，再在看最终性能之前冻结并独立核验fresh接入；随后一次完整矩阵据实判定。当前仿真终点保持完整，实时与真机另需各自证据。[036:44](../plans/v0.0.1/036_project_goal_20261023.md#L44)
