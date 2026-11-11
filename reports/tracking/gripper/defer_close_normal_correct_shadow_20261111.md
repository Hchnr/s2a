# 原夹爪控制器的正确参考影子结果

2026-11-11。按[登记](../../../plans/v0.0.1/tracking/gripper/defer_close_normal_correct_shadow_20261111.md)，
复用原`LoadedVelocityControl`夹爪与修正运算顺序的机械臂候选之18条已独立审核
正确参考，覆盖全部4907周期。原暂缓闭合规则只在**correct_08、step85**提议
`+1 → 0`；当时原轨迹已抓取`cream_cheese_1`，原执行后的opening误差为
−0.002770。全部提案与独立标量几何及原预览逐项核对，最大数值差`1.39e-17`。
这是对原轨迹的影子检查，不能证明干预后的抓取和任务结果。

该case的原夹爪控制器与此前带`.35`释放下限控制器虽为不同实现，但只读比较发现
两条原case08轨迹的182条完整动作、夹爪决策、对象及任务标签均一致；23个chunk
文件和包含183个仿真端点的轨迹数组文件分别具有相同SHA。因而此前[受载点单动作
物理对照](defer_close_correct_feedback_20261111.md)也对应原夹爪控制器的同一
历史状态与原始提案；其干预后两条历史命令仍是固定动作，不是该控制器的在线
反馈或长期抓取证明。其余17条轨迹未作上述逐文件等价断言。

task1短窗使用同一原夹爪基线和暂缓规则，但机械臂采用质量约束反馈；本矩阵使用
修正运算顺序的原机械臂候选，两者尚不是同一完整控制合同。18条历史合格不能
授予新暂缓规则或task1反馈整体资格；仍须以统一候选完成18条实际正确参考、
12条WAM开发及24条新初态验收。

18个影子worker及dry/执行子进程均实际wait/reap、exit0、无信号。32进程CPU
专项19项通过、0跳过，全仓Ruff及格式检查通过；没有新物理、模型调用或GPU。
首次CPU准入因根文件指纹漂移在收集前失败，原字节已恢复，失败收据及独立新QA
路径见[修复记录](../../../plans/v0.0.1/tracking/gripper/defer_close_normal_shadow_qa_repair_20261111.md)。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/defer_close_normal_correct_shadow_20261111/result.json` | `dbe953795285ae9a75943afe8dd7964f314d750fbe33c60104353fee637819c6` |
| `outputs/defer_close_normal_correct_shadow_20261111/workers.json` | `bb4ff0dfc31397db45e35bc419e958f2aaf76b0fe11670da30d4f22f6f3b2208` |
| `outputs/defer_close_normal_correct_shadow_dry_20261111/result.json` | `0b3616d55270aada49003c3a86f064ebee443ee5309d157a49d60e2d29eb4ed3` |
| `outputs/defer_close_normal_correct_shadow_supervisor_20261111/result.json` | `d618ccd7fa7a4b21f455aedb081e95e45b1a9069c34222286e011ad271e00f44` |
| `outputs/qa/defer_close_normal_correct_shadow_v2_20261111/result.json` | `4e42b922863c2c109b4d341601fbb52e7bebc086a5cde11bac6724f763184676` |
| `outputs/tracking_correct_reference_ordered_20261027/case08_candidate/steps.json` | `393f198d687a42dea0cd50dcc4d6965b54c534d1c66bb1ed963af19ca3d94229` |
| `outputs/release_floor_correct_reference_20261028/case08_candidate/steps.json` | `1e83d16cdea327542e15b84bf1ce16f1c186ae4fcf1747c19962e420e846312a` |
| `outputs/tracking_correct_reference_ordered_20261027/case08_candidate/chunks.json` | `6f150a644c0cb0d063f29e7119c6084385f0c6a5d6cdc9633f01b367f1ada3ef` |
| `outputs/release_floor_correct_reference_20261028/case08_candidate/chunks.json` | `6f150a644c0cb0d063f29e7119c6084385f0c6a5d6cdc9633f01b367f1ada3ef` |
| `outputs/tracking_correct_reference_ordered_20261027/case08_candidate/actual_trajectory.npz` | `e97c01a73c2d41306b29e6252ef1b3c9be4e128a65683b6ea85b6230e00c6e58` |
| `outputs/release_floor_correct_reference_20261028/case08_candidate/actual_trajectory.npz` | `e97c01a73c2d41306b29e6252ef1b3c9be4e128a65683b6ea85b6230e00c6e58` |
