# 022：当前8400的固定任务描述配对诊断

固定长任务描述**未通过事前效果门槛，不推广物理或fresh评测**。真实96次生成及全部来源、数值和释放检查通过：前48次完整复现原短任务输出，后48次只改变实际caption的command；旋转误差小幅下降，但H16位置和opening误差退化，未达到预定改善要求。

本项以`e117243`登记，启动前以`d4277dd`补充空`task_plan: []`的条件隔离合同。使用[登记配置](../configs/wam_static_command_reproduction_20261021.json)中的8400非EMA权重（`iter_000008400/model`），固定active10 mask、H40、20Hz、35步、guidance1、shift5、SDPA；模型仅物理GPU7/端口10047，CPU显式空CUDA/LD。既有六个开发episode共24阶段样本、22不同anchor，每种文本各seed42/43，共48请求；其中44个不同anchor/seed组合，重复样本和重叠窗口不算独立episode。H8/H16/H40每条件分别计384/768/1778个有效未来端点，不混用018共同裁剪后的1398端点。

每任务只使用原manifest首次出现样本的published H40 caption中的command，所有该任务episode、阶段和seed沿用同一句长描述。实际stock准备后仅补同baseline已有的空plan占位；非空示范plan和current_subtask均拒绝。启动前48对真实CPU准备验证了caption除command外的字典、键序、像素、完整packed video、domain/mask/fps/shape一致。运行时另存actual prepare/predict账本，逐对核验实际caption替换command后的标准JSON字符串精确一致，并与response的conditioning-caption SHA绑定；不以请求文本代替实际模型输入证据。

前48次reproduction的**全部40×128 raw及normalized值逐值精确等于旧8400/g1缓存**，通过准入后才执行expanded48次。独立审计`69dff45`检查全部96条的seed、权重、mask、时间戳和反归一化；使用冻结的011独立Gram–Schmidt/SciPy积分，从当前观测pose积分local delta并对照真实未来state，完全绕过S2A。原逐endpoint误差及hold/历史匀速外推指标也逐项核对。

下表每格为 **位置RMSE mm / SO(3) RMSE rad / opening MAE**；opening为归一化开度，所有方法使用相同有效端点。

| 方法 | H8（0.4秒） | H16（0.8秒） | H40（2秒） |
| --- | --- | --- | --- |
| 原短任务（48次精确复现） | 24.384 / 0.088644 / 0.058538 | 44.623 / 0.177223 / 0.086038 | 85.521 / 0.479894 / 0.114660 |
| 固定长command | 24.371 / 0.086226 / 0.056215 | 46.298 / 0.171795 / 0.088021 | 88.331 / 0.472225 / 0.115813 |
| 保持当前state | 38.201 / 0.052695 / 0.076542 | 71.143 / 0.114498 / 0.161877 | 128.832 / 0.309419 / 0.255167 |
| 历史匀速外推 | 26.426 / 0.043213 / 0.076542 | 68.558 / 0.107911 / 0.161877 | 179.117 / 0.327273 / 0.255167 |

冻结门槛逐项判定，未追加数值余量或据结果修改描述：

- H8三指标均不退化；H16位置及opening退化，六项共同不退化要求失败。两个短窗口均没有任何一项改善10%。
- H40位置误差超过原值的1.02倍，H40保护门槛失败。
- 用每endpoint的`(position/0.01m)^2 + (SO3/0.05rad)^2 + (opening/0.1)^2`求均值，三任务只有cream的H8/H16均不退化；六episode只有1098、898两条的两个短窗口均在原1.02倍内，未达到至少5/6。完整task/episode/phase指标保留在最终产物。

因此`eligible_for_separately_registered_development_physics=false`。这一个固定静态长描述没有提供足够改善，不能将当前失败主要归因于短任务文字，也不据本项宣布任何闭环任务成功或原checkpoint正确。结果限于已查看开发episode；训练未见、fresh泛化及真机均未在本项验证。单条示范未来state也并非任务唯一正确轨迹。

来源与退出审计同时通过：852个输入SHA在全量审计前后及报告发布检查时一致；报告另从已绑定逐endpoint误差独立重算全部pooled指标，最大差0。96个唯一生成目录、96对prepare/predict账本及同一owned Uvicorn生命周期内96次POST成功全部覆盖。两份phase快照各含coordinator/model/query三身份，共享同一coordinator与model，合计**四个历史PID/birth身份，非六进程同时ready**；两个query worker均returncode0并reaped，模型受控SIGTERM退出（-15），四个登记进程均已退出。审计器7项针对性CPU测试、独立源码审查和Ruff通过。实验新增96次模型生成、0物理、0拟合、0fresh；独立审计及报告阶段新增模型请求/物理步均为0。

`prepared.json`（`44cfd712…`）和`prepared_v2.json`（`42921322…`）只做过CPU准备，原件保留；实际唯一模型运行绑定`prepared_v3.json`，明确纳入启动前源码更新及空plan补充，不将元数据修订当作性能重试。源码依据：[实际caption配对](../scripts/audit_wam_static_command.py#L34)、[raw/norm合同](../scripts/audit_wam_static_command.py#L57)、[逐case state重算](../scripts/audit_wam_static_command.py#L130)、[固定门槛](../scripts/audit_wam_static_command.py#L335)、[独立退出审计](../scripts/audit_wam_static_command.py#L474)。

| 证据（相对`outputs/wam_stability_20261021/`） | SHA256 |
| --- | --- |
| `static_command_state/prepared_v3.json` | `5f2159d3cf8704006ed9f4fe87480a4f181caabccd0f41c1cd9b8576efd86ad0` |
| `static_command_preflight/actual_cpu_preparation_v3/result.json` | `5e71bd6e19eae8902946dee9c3df26222b54331908e734d41e522ea18538db77` |
| `static_command_preflight/root_launch.json` | `11c0526caa9730ea22f23335f5da14a485adff02fa8cded1a3c30dfa2c38b760` |
| `static_command_preflight/reproduction_actual_ready.json` | `46b834caacb7325996a0fd0c43a2a07d1b091655233992a8f9e811ed5ccc5fa1` |
| `static_command_preflight/expanded_actual_ready.json` | `c26dbc41b40ebea8b3ce2e7b8d576707286d8ac701f9c2b8690700e1aeb5859d` |
| `static_command_review/auditor_69dff45_independent_peer.json` | `3807a4f24446f7f9a1f76861e27a6b684313057687c4ea0cb052f1662e76b28d` |
| `static_command_review/complete96_verified.json` | `010e404aa1df40b2daaaeab0e93675e51029f8c1503e32cf11fd446bcc40c456` |
