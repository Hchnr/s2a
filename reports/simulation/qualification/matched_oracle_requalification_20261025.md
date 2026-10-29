# 062 当前环境匹配oracle正控

固定三任务旧init0/seed42、position_hold、完整旋转、无governor、H40、每8步重观测、600步/20Hz/末40步原生稳定；与061相同控制器和参数。预算1800物理步、0模型/拟合/fresh，正确参考不计世界模型成功。计划见`plans/v0.0.1/062_matched_oracle_requalification_20261025.md`。

`run_matched_oracle_case.py`复用固定run_position_case及已验证OpenCV恢复，三个实际Python3.11 dry均通过；Ruff通过。首启动在监督器线程准入阶段拒绝，0物理/模型；原目录`outputs/matched_oracle_requalification_20261025`保留source_sha256及admission_failure.json。线程按监督器固定三项8修正，新目录`outputs/matched_oracle_requalification_20261025_v2`，控制算法未变。

持久化owner PID3388076/birth582527784，首client3388084/birth582527813，实际启动后按/proc复核。所有执行源码及计划配置已在source_sha256.json绑定；后续独立检查器单独新增，不修改在运行源码。CPU/OSMesa显式空CUDA/LD，0模型任务。

首task0已完整600步/75段，producer complete=True/error=None/stable_success=True；位置RMSE0.0018380899154764902m、旋转RMSE0.003339389270577068rad，数值与历史oracle一致。实际stage passed=True，原client已退出。第二task1 client PID3389762按launch及/proc确认同birth存活，已落盘360步。owner仍为原进程，固定三条继续。

目前仅首条完整，不能认证整个062通过；新增`scripts/verify_matched_oracle.py`已通过Ruff，待全三条正常完成后实际调用原source_proof和verify_case(full=True)，检查全部机械/几何/夹爪/参考、退出收据及原10mm/0.05rad门槛。尚无新的WAM/fresh性能结论，总目标未完成。

## 最终独立资格通过

实际全量审查进程正常exit0，`independent_verified.json` SHA c32ce32a4c6bcc4d82c84094aeac12ca2917f06e6c24c02c62264942a5e1cce4，complete/qualified=True，232消费输入前后复核；启动source_sha256全表再次核验无漂移。三条均600步/75段，独立重算native末40稳定3/3；task0/1/2成功步531/458/501，尾连续531/454/501。位置RMSE分别1.838090/1.421041/1.547627mm，SO(3)RMSE0.003339/0.006744/0.002595rad，均通过原10mm/0.05rad门槛。全部实际wait0/reap且未发终止信号，固定1800物理步、0模型/fresh。

当前资产环境中，正确参考经固定S2A能够完成同三任务旧初态；这排除了“该接口无论何种参考都不能完成任务”的解释，不证明其能执行任意模型参考，也不证明WAM或fresh成功。061候选0/6仍保留，下一步重点检查生成参考的接近/抓取阶段与训练锚点覆盖，暂不盲目增益调参或启用覆盖不足的打开阶段保留组件。
