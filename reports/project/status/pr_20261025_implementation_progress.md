# PR 信息 / PR information

## 中文

**标题：** feat: 完善 WAM 状态轨迹验证、S2A 控制与可审计实验流程

**说明：**

本次提交收录上次提交之后的 WAM 状态参考训练与同步验证、S2A 控制器及原生数据实验实现，并保留成功、失败和未执行项。目标是在固定机器人和控制接口下，将期望状态轨迹转换为可执行 action，以实际前向状态误差和任务稳定性评价效果。

- 增加状态参考采样、轨迹目标、末节点修正、同步执行及独立验证。
- 增加夹爪状态与意图控制、参考预览、响应辨识、有限原生命令规划，以及平移和位置/速度逆控制对照。
- 增加持久化实验 owner、实际 wait/reap 凭据、来源指纹、独立机械/标签/图像核验及相应配置、计划和报告。
- 记录路径迁移与恢复依据，遵守 CPU/OSMesa 和允许 GPU UUID 的执行限制；生成数据、权重及本地环境不纳入 Git。

**验证：** 全仓 `ruff check` 和 `ruff format --check` 通过；提交前复跑当前控制链相关 51 项测试通过。未在本次提交前运行全仓 pytest。083 冻结的 699 份来源指纹复核一致。

**结果与限制：** 082 完成 18 条、4907 步及全部生命周期和独立审查，任务稳定 15/18，综合达标 14/18，未达到原 18/18 门槛。083 提交时仍在运行，前 7 条独立核验完成、6 条达标，case03 任务失败，已不能通过本轮 18/18 资格。上述正确参考结果不是 WAM 闭环成功；WAM 开发及留出新初态验收尚未通过，总目标未完成。

## English

**Title:** feat: extend WAM state-trajectory validation, S2A control, and auditable experiments

**Description:**

This change collects WAM state-reference training and synchronous validation, S2A controllers, and native-data experiments added since the previous commit. Plans and reports distinguish successful, failed, and unexecuted work. The objective remains to convert desired state trajectories into executable actions under a fixed robot and controller interface, evaluated by forward state error and task stability.

- Add state-reference sampling, trajectory objectives, terminal corrections, synchronous execution, and independent verification.
- Add gripper state and intent control, reference previews, response identification, finite native-command planning, and translation and position/velocity inverse-control comparisons.
- Add durable experiment ownership, actual wait/reap receipts, source fingerprints, independent mechanical/label/image checks, and explicit configurations and reports.
- Preserve relocation and recovery evidence and CPU/OSMesa and allowed-GPU-UUID restrictions. Generated data, weights, and local environments remain outside Git.

**Validation:** Repository-wide Ruff lint and format checks passed. The 51 tests covering the current control chain passed again before commit. The full repository pytest suite was not rerun for this commit. All 699 frozen source fingerprints for experiment 083 remained unchanged.

**Results and limitations:** Experiment 082 completed 18 trials and 4,907 steps with independent and lifecycle checks: 15/18 stable, 14/18 meeting all performance gates, below the required 18/18. Experiment 083 is still running at submission; its first seven trials were independently verified and six qualified. Its case03 task failure already prevents passing the 18/18 qualification. Correct-reference results do not establish WAM closed-loop success. WAM development and held-out-initial-state acceptance remain unmet; the overall objective is incomplete.
