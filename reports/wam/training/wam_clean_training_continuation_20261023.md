# 041有限训练接续：完整D0与W1前置失败 — 2026-10-23

**新D0已完整通过，随后W1在任何拟合前失败，041 supervisor已停止。** D0为300请求/10500节点，producer与独立审查通过，模型实际wait0/reap、同birth、无signal及GPU释放；不是任务成功。W1为`load_context`将执行验证器源码ref交给outputs-only产物reader，在创建fit输出、读取未来标签、fit和QR之前拒绝，实际均0。保留原False，不重启旧supervisor。总目标仍为原预留24条稳定≥20/24、各任务≥4/8。[当前总览](status_current_20261023.md)

实际结果：[D0 producer](../outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/capture/D0/result.json)，SHA `d23b31eb9e40f9f3e8f22571399d164d8980ba3f4d75b10876d740edd52b7dae`；[D0独立审查](../outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/review/D0_verified.json)，SHA `5d62df6fae3aa425a74e02046b638a4543eb95b7fa31eda934b1fdad58d8b08d`；[D0 owner](../outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/orchestration/D0/result.json)，SHA `8ce9a076828aab59dfd1478bbccfef1160f4019e9b30d10ad091eea518364950`。完整capture审查沿登记边界核10500点；vision/flat/head省略字段只hash并沿用online/037＋039资格，不假称重新核了省略字段。

W1 producer PID1555973/birth568340588实际wait1、reaped且无signal；stage owner1555903亦正常wait1/reap。主[supervisor False](../outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/supervisor/result.json)保留，D1/W2/STATE/DEVELOPMENT没有启动。失败精确来源为[fit contract:171](../src/s2a/clean_training_fit_contract.py#L171)→[proof source callback:58](../src/s2a/clean_execution_proof.py#L58)→[outputs限制:77](../src/s2a/clean_training_fit_contract.py#L77)。[实际失败审计](../outputs/clean_training_postcapture_repair_20261023/incident/audit.json)，SHA `38486bfc13c563886642757c75fee2de6c605bbb18c299f88a94d0a8bf93df7e`。

[D0独立只读复核](../outputs/wam_fresh_acceptance_implementation_20261023/d0_data_readonly_audit_20261023_v1.json)，SHA `2a15358b0007e1897e12a37fab6de907471d486625a7ff69b9f99d807fcf027c`，支持保留并在原冻结合同下复用D0；0新增数组decode，不授权重跑fit。后续[043](../plans/v0.0.1/043_clean_training_postcapture_20261023.md)仅做另行准入的有限W1 CPU接续，原484来源、484快照和bundle不改。下列启动与存活段均为此前时刻快照。

06:43:34 UTC进度快照：新D0已保存**8/300个完整请求、280/10500个节点**，worker进度与8份complete ledger一致；三个实际PID/birth仍匹配，supervisor为PPID1，484源无漂移，允许UUID的GPU实际工作。它仍在运行，尚无整轮退出、GPU释放或完整capture独立资格；不是8次任务成功。[进度观察](../outputs/clean_training_continuation_final_qa_20261023_v2/progress_observation_v1.json)，SHA `a525e1ad8249a8acf3275c6245d4c0311c39047f60513a39b5cc49ff9d48cd65`。

原D0的20完整请求与25个partial节点、缺失退出资格和未知中断原因保持原记录，不用于新fit，不覆盖或拼接旧目录。[中断报告](wam_clean_training_interruption_20261023.md)与[041运行前协议](../plans/v0.0.1/041_clean_training_continuation_20261023.md)分别保存现场和新一次有限接续的范围。

新prepared仅改变原metadata的输出root。独立审查核4312份消费文件、旧462源及462快照、1661旧产物、11个bundle文件与34处有限metadata变更；300个实际31位seed保持等于040。metadata审查不重新执行原数据资格，也未解码future数组。完整执行审查另核484源码与484快照、累计预算、精确命令接口、归一化/35节点sampler、控制合同、当前资源及最终QA，共5211份消费文件。[独立metadata](../outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/metadata_verified.json)、[独立执行审查](../outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/execution/verified.json)

| 资格文件 | SHA256 |
| --- | --- |
| 新metadata verified | `1f3441a2369c991ff89ff88bb85fb796cce59a75131e0db4682d030e6cac886c` |
| 未授权proposal | `e3bba612e0bae749d4a54410fa743bf0f5a63fad18fc201dca5afaad31248b0c` |
| 独立execution verified | `bdb51d79909215e75b08488ef4f77bcf599efb1b8ee5b26b7be972c35b5cf5ee` |
| admitted | `50e0a03d429b43343b5ff809879bf46bfeeff4d7b87bf7f2d70cb488d7a48d31` |
| 实际seed-plan | `5ec7c51868d462cae22e2d24c6b5e8a39f9755d722844651bd290f271f2cdc2c` |

[admitted](../outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/execution/admitted.json)允许新根内固定D0→W1→D1→W2→STATE→DEVELOPMENT各阶段一次，禁止自动重试和fresh性能。新预算1599请求/55965 denoise/4模型初始化；连同旧中断运行保守累计≤1620请求尝试、≤56700 denoise、≤5模型初始化，生产fit2/独立QR2/物理7200不变。原20条未获完整资格，不进入任何新fit。

最终相关QA为23个模块**730 passed、1 skipped**，141.91秒；Ruff check及format --check通过，466主Python文件，完整484源在检查前后字节相同。[QA收据](../outputs/clean_training_continuation_final_qa_20261023_v2/qa.json)，SHA `8800088bd8e424c650ac96c3523476763edd0c2aec113cb7187ccedc059aac83`。skip是[旧真实环境fixture探针](../tests/test_native_environment.py#L112)路径不存在，另有已合格的当前cv2导入保护；不算通过，也不声称全仓pytest已绿。

此前v1的727/1及其[集成问题记录](../outputs/clean_training_continuation_final_qa_20261023/integration_review.json)保留：真实bundle owner缺少`REPO/scripts`导入路径。新supervisor显式提供`REPO/src:REPO/scripts`并记录实际子进程环境，不继承任意调用者路径；新增真实独立进程检查覆盖10个已发布入口及metadata-only D0 dry拒绝，禁止模型/物理导入和数组解码。[环境实现](../src/s2a/durable_stage_supervisor.py#L107)、[实际集成检查](../tests/test_durable_stage_supervisor.py#L303)。旧462源、已发布bundle及独立checker没有因修复而改动。

admitted发布后，真实D0 `--dry-run`以新prepared/admitted的精确SHA成功，实际worker argv指新bundle collector、固定35节点和唯一允许UUID，不加载模型。[dry收据](../outputs/clean_training_continuation_final_qa_20261023_v2/d0_admitted_dry_run.json)。随后短`--launch`真实exit0返回，持久化supervisor继续存活。[launch收据](../outputs/clean_training_continuation_final_qa_20261023_v2/detached_launch_cli.json)

06:40:47 UTC实际观察：supervisor PID1390098/birth567832941、PPID1、PGID/SID均为自身；D0 owner1390101/birth567832997由其持有；模型worker1390170/birth567833132由D0 owner持有。三者birth均与真实launch收据一致，启动器退出后仍存活。CPU supervisor/owner空CUDA/LD、三线程参数8及显式import路径；模型仅`GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f`，驱动对应visible2/PCI BA/minor6，未使用另一允许卡。GPU开始占用，当前484源及原462源均无漂移。[实际启动观察](../outputs/clean_training_continuation_final_qa_20261023_v2/launch_observation_v1.json)

supervisor持有各stage owner的真实Popen，串行检查退出与完整owner结果，异常仅清理自有child并停止；owner继续持有模型与独立审查的Popen。CPU检查和本次PPID1观察支持普通launcher退出后的独立存活，不证明抵抗容器销毁、SIGKILL或外部终止。任何未完成阶段仍须其真实wait/reap、正常GPU释放及完整独立审查，不能提前认证成功。

固定开发任务稳定门及相对new raw的收益门分别报告，均不能代替最终24条；当前没有新候选收益或新初态性能结论。完整最终接入还缺统一候选、fresh首相机witness与native/oracle/WAM矩阵，[差距审查](wam_final_acceptance_gap_audit_20261023.md)。
