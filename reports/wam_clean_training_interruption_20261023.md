# D0中断审计与独立接续准备 — 2026-10-23

更新：新041现已独立准入并实际启动，见[接续报告](wam_clean_training_continuation_20261023.md)。以下保留中断核查及接续准备时刻的记录；原D0缺失的退出资格不因新运行而改变。

**原D0已停止且不合格；041独立接续尚在实现/准入准备，未启动新的模型。** 总目标仍未达到，最终24条性能未运行。本报告更正此前D0“运行中”的启动快照，不将缺失的退出证据补写为正常。

2026-10-23 05:57 UTC重查时，会话4754已不可访问，owner1292033/birth567473266与worker1292382/birth567478090均无对应进程。独立复核无存活命令副本，允许UUID的GPU已实际memory0/util0/apps空。原owner/capture/审查result、worker exit/release及actual wait/reap证据均缺。中断原因及signal未知；turn_aborted只与时间相近，不能证明是谁或以何信号终止。[独立事件审计](../outputs/clean_training_interruption_audit_20261023/audit.json)

现场保留20个完整ledger/700点，以及第21请求25个已持久化点；已完成调用下界725，精确实际调用总数未知。按尚未完成的35-call scope保守计725..735，不将保存点数当成精确总调用数。原462源与462快照未变，原1661个产物/297900896 bytes已逐文件SHA登记，未读native/validation future数组，也未进行fit或任务。[原产物清单](../outputs/clean_training_recovery_design_20261023/interrupted_artifact_inventory.json)

原20条和partial25不进入训练，也不移动或覆盖原路径。新的[041有限接续协议](../plans/v0.0.1/041_clean_training_continuation_20261023.md)另设`recovery_20261023`根，拟完整执行同一300输入/seed的D0及原后续阶段一次。新单轮预算仍1599请求/55965 denoise/4模型初始化；连同旧中断运行保守累计≤1620请求尝试、≤56700 denoise、≤5模型初始化，生产fit2/独立QR2/物理7200不变。原038/040结果和预算保留，不假装未消耗旧运行，也没有自动重试。

新bundle从040冻结字节生成，仅更改明确的metadata/root/import/实际入口路径，独立checker重新核变更；采样、C0/残差、fit/独立数学、控制及评分不变。新prepared只改原b55的root字段，独立验证派生；新seed plan引用新prepared SHA但300实际seed不变。原20未获完整生命周期资格，不通过拼接绕过原门。

新supervisor以独立session、文件日志及真实Popen持有各stage owner；短launch退出后仍存活，并保存真实wait/reap/timeout。CPU小实验已证普通父退出后的存活、正常完成和仅自有child的超时回收；不能证明抵抗容器销毁或SIGKILL。[设计及CPU实验](../outputs/clean_training_recovery_design_20261023/review.json)

下一项是新metadata/bundle、独立metadata审查、最终23模块QA/Ruff及完整source冻结，再经未授权proposal→独立审查→admitted，才允许唯一detached启动。当前不具备新的D0完成、拟合、开发收益或fresh证据。最终验收还缺统一候选、fresh首相机witness与完整native/oracle/WAM矩阵，见[最终验收差距审查](wam_final_acceptance_gap_audit_20261023.md)。
