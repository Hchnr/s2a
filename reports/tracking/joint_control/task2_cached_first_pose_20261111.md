# 已评估双点搜索中的首点候选

2026-11-11。按[预登记](../../../plans/v0.0.1/tracking/joint_control/task2_cached_first_pose_20261111.md)，
在[因果回退反馈](task2_causal_shift_fallback_20261111.md)独立重放后的raw step33和
terminal058 step85停止状态，分别以已保存LP初值及已完成计划后移初值重建guarded
双点搜索。隔离记录只读取原搜索已经评估的rollout及原安全筛选，不增加模型响应。

| 开发停止状态 | 初值 | 原双点搜索响应 | 已评估线性安全计划 | 首点双阈值候选 | 独立首点审核 |
| --- | --- | ---: | ---: | ---: | --- |
| raw step33 | 已保存LP初值 | 148 | 1 | 0 | 未尝试 |
| raw step33 | 后移已完成计划 | 44 | 1 | 0 | 未尝试 |
| terminal058 step85 | 已保存LP初值 | 31 | 6 | 0 | 未尝试 |
| terminal058 step85 | 后移已完成计划 | 68 | 1 | 0 | 未尝试 |

四次重建搜索共291次模型响应，按各次搜索分别去重的9个线性安全计划均没有同时满足首点位置
≤0.01 m及SO(3)≤0.05 rad（沿用`1e-10`审核容差）；也没有双点候选。由于没有
首点候选，0次新独立候选审核，不能把线性安全筛选写成独立物理安全。两个初值的
共同关节仿射域最大差分别约`2.67e-15`和`1.78e-15`，终端方向均为空。

这排除了**本次重建搜索已评估缓存**中的首点回退，不证明原在线求解的完整缓存、
其他搜索点、冻结模型全局或真实物理不可达。原在线停止求解的原始细化种子并未
完整保存；其响应为98/49，本次已保存LP初值重建的响应为148/31，不能声称重放
了原求解。不能据此直接增加派发动作或改变原任务成功门槛。

## 复核与费用

合成测试核对隔离记录与原求解的结果逐值一致、响应次数不变，并验证只有首点通过
且独立八周期安全审核通过的候选才能被标记为合格。实际使用2个静态模型、2次
`mj_forward`、4次准备rollout和4次搜索；2个worker与监督进程实际wait/reap、exit0、
未发信号，源码及输入前后一致。63项32并发CPU专项、全仓Ruff/格式和零模型dry
通过；本轮没有新物理积分、新WAM、GPU或最终留出样本。

下一步应继续检查停止前的因果状态与参考衔接，或以固定预算开发新的候选搜索；
首点缺口、实际完整8周期与任务成功仍须在自己的物理反馈和独立重放中验证。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/task2_cached_first_pose_20261111/result.json` | `f1752184a8f20957303e49b655020b1d95387aedc089df328fba58ac858ff898` |
| `outputs/task2_cached_first_pose_20261111/workers.json` | `4cb036365ef3ee5cb780b057c175be29e903cb38791808566c8bd6119a8a06b1` |
| `outputs/task2_cached_first_pose_dry_20261111/result.json` | `22c5b2fcb1b6c8d4373cf40f0a53ed49cc664d0ab7cf5b6493c91cad10e3fab4` |
| `outputs/qa/task2_cached_first_pose_20261111/result.json` | `245eb9be46f7921dc692bbebeeadcb6072e296190be792d13e9d0f518118bb7e` |
| `outputs/task2_cached_first_pose_supervisor_20261111/owner_exit.json` | `db19bbe71472b24a27f217dfb9e6c42333cbd4b3b031cb53e4651729434bbca0` |
