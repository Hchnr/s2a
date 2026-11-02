# 040执行准入与D0启动 — 2026-10-23

后续更新：原D0进程已消失且缺正常退出资格；保留20完整请求及第21条25节点，新的041独立接续仍在准备。见[中断审计与接续准备](wam_clean_training_interruption_20261023.md)。以下保留05:44 UTC的启动快照，不能作为当前仍在运行或已完成的证据。

**038的完整执行入口、最终QA和独立准入已通过，D0唯一模型进程已启动；本快照日志记录8/300请求、280/10500 denoise。尚无整轮D0资格、真实W1/W2拟合、开发任务或fresh收益。** 总目标仍是固定机器人/控制接口下可靠跟踪生成参考并完成留出任务；最终12新初态×42/43的24条中稳定≥20/24、各task≥4/8保持不变。[总目标与标准](../plans/v0.0.1/036_project_goal_20261023.md#L7)

## 已完成的执行准备

补齐99生成STATE worker/owner/独立评分、单实例开发私服、固定12case物理wrapper及独立控制/因果计分。三zero、48配对、全部节点及生命周期各自核验；state门与任务门分列。服务只有完成12/900才可正常finish；cleanup错误实际CPU子进程测试已证先保存False再exit1。独立开发审查不冒称额外运行过历史003回归。[040执行合同](../plans/v0.0.1/040_clean_training_execution_20261023.md)、[cleanup修复核验](../outputs/clean_training_development_qa_20261023/integration_fix_verification.json)

最终源码版本上实际运行19个相关模块：**613 passed、1 skipped，89.88秒**；全仓Ruff check及457个Python文件format check通过，462项源码/配置/计划SHA测试前后一致。这是相关软件QA，不是全仓pytest均通过。跳过项为旧native import-only测试所需observer路径缺失；当前精确034 probe与cv2 guard/实际控制依赖的复用证据另经补充预检和独立040审查核验，仍为有条件复用，未新增物理正控。[QA与实际命令](../outputs/clean_training_execution_final_qa_20261023/qa.json)、[跳过位置](../tests/test_native_environment.py#L112)、[控制补充](../outputs/clean_training_development_preflight_20261023/dependency_addendum.json)

producer生成未授权proposal；独立stdlib checker重核全部462源/快照、3028个消费文件SHA、034→037→039原资格与原False、normalization31/proj0、35个实际sigma及控制guard、资源与实际QA后，给出`eligible_for_execution=True`。最后admitted精确绑定proposal/审查SHA，三处运行准入均核完整证明链。原b55 metadata永久保持`ready_for_execution=False`。[准备器](../scripts/prepare_wam_clean_training_execution.py#L231)、[独立审查](../scripts/verify_wam_clean_training_execution.py#L1066)、[运行准入](../src/s2a/clean_training_runtime.py#L93)

准备时实际资源为磁盘324801519419392 bytes、有效RAM922801786880 bytes、affinity192核；独立checker再次观察资源。允许UUID为`GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f`，PCI `00000000:BA:00.0`/minor6，启动前显存0/apps空/利用率0，31338/31339可绑定。实际owner仍持UUID锁并再次检查；此处不是预占或峰值内存测量。[资源收据](../outputs/wam_stability_20261021/clean_residual_training_20261023/execution/resources.json)

## 运行前发现并修正的seed合同

原038的300个63位seed虽然元数据完整，却全部被冻结服务的31位请求schema及32位NumPy `RandomState`拒绝。还发现renderer按实际seed决定空`task_plan`或缺键，不能用42/43的空[]分支代表所有训练请求。这些问题均在首次038模型调用前查明；未改冻结Cosmos、随机数实现或原b55字节。[真实模型解释器CPU探测](../outputs/clean_training_execution_verification_20261023/seed_rng_probe.json)

040预先登记新的完整SHA域派生31位seed方案，逐条绑定原anchor/root/63位身份及新payload SHA；300唯一、排除42/43，无冲突重抽。原canonical/identity仍只表示原source身份。执行metadata是显式副本，只替换两个request seed、noise roster及算法说明并增加seed-plan ref；原文件不回写。模型、采集审查及fit三门使用同一实际映射。独立checker另行重建全部300项；全部新seed在真实解释器的CPU schema/NumPy/Torch探测通过，其中25条caption确实无task_plan。采集与独立checker严格核其真实分支，不补写caption或改变词元。[预注册修订](../plans/v0.0.1/040_clean_training_execution_20261023.md#L5)、[纯metadata映射](../src/s2a/clean_training_seed_plan.py#L44)、[独立映射探测](../outputs/clean_training_execution_verification_20261023/seed_plan_independent_probe.json)

| 不可变收据 | SHA256 |
| --- | --- |
| [原metadata b55](../outputs/wam_stability_20261021/clean_residual_training_20261023/preparation/prepared.json) | `b55c214ffec328ee94f2074007c15dfe62486512aeb660b251fdb21f38e970f7` |
| [最终QA](../outputs/clean_training_execution_final_qa_20261023/qa.json) | `d64ac8c19a808f767bdea60bf82fca7057d198559f496062415e3453fa9e072c` |
| [31位seed plan](../outputs/wam_stability_20261021/clean_residual_training_20261023/execution/seed_plan.json) | `d33aaf103cb00ae01f4e725709357b54da0f2dafae8102421c241c2627fe293d` |
| [未授权proposal](../outputs/wam_stability_20261021/clean_residual_training_20261023/execution/proposal.json) | `6fc891c038ca8beeda2b9d2a384a661395691c1e3e6b79bafd4c6f42a3aef142` |
| [独立执行审查](../outputs/wam_stability_20261021/clean_residual_training_20261023/execution/verified.json) | `702033dd0271897affc372e5c95b0e75a23a623916195b682f78e0ef77d20267` |
| [admitted](../outputs/wam_stability_20261021/clean_residual_training_20261023/execution/admitted.json) | `7f4d099f65e6bd98fa671f1d76e35fe9f68bc8fc411718e67d0590422271033b` |

## 实际启动及剩余工作

D0 dry-run实际exit0，随后唯一owner于2026-10-23 05:40:27 UTC启动（PID1292033/birth567473266），持有模型worker PID1292382/birth567478090。worker命令明确roundD0与上述单一UUID，原S2A cwd、空LD，CPU owner空CUDA，模型只见获准UUID。live快照记录8/300请求、280/10500 denoise；首请求实际seed839804476与plan一致，35节点ledger/trace已落盘，462源码SHA仍一致，原b55 SHA未变。此为运行中日志与首份产物观察，**没有正常退出/释放或完整独立采集资格**。[dry-run](../outputs/clean_training_execution_final_qa_20261023/d0_dry_run.json)、[实际owner launch](../outputs/clean_training_execution_final_qa_20261023/d0_owner_launch.json)、[实际model launch](../outputs/wam_stability_20261021/clean_residual_training_20261023/orchestration/D0/model/launch.json)、[8请求live快照](../outputs/clean_training_execution_observation_20261023/d0_progress_v2.json)

完整协议仍为D0 raw300/10500→W1一次420000行fit/独立QR→D1同300输入/初噪声的固定W1→union W2一次840000行fit/独立QR→99完整生成诊断→raw6/W2六开发任务。全部上限1599请求、55965 denoise、4模型初始化、生产2fit＋独立2QR、7200物理步，无重试。采集独立完整核验及GPU正常释放前不解码真实训练future标签。W1只是中间策略，W2唯一候选；没有本轮成功率可报告。

D0是长任务，其完整结果以新的owner/独立采集收据为准，本启动快照不能代替完成证明。保留旧QA、失败与metadata记录，不覆盖已有产物。最终fresh24、实时与真机均未运行，总目标持续进行。
