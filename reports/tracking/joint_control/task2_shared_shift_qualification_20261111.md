# 组合反馈停止处的已完成计划资格

2026-11-11。按[预登记](../../../plans/v0.0.1/tracking/joint_control/task2_shared_shift_qualification_20261111.md)，
在[组合反馈](task2_margin_guarded_feedback_20261111.md)已独立重放的两个停止状态，
各取该分支最后一条实际完成的八步计划后移一拍，固定当前实测状态、上一实际动作、
已完成误差与已生成参考，检查冻结模型质量和完整安全。没有重新优化或派发。

| 开发停止状态 | 首点位置 / SO(3) | 第二点位置 / SO(3) | 首点模型候选 | 双点模型候选 | 后六点最坏归一化缺口 | 独立模型安全 |
| --- | --- | --- | --- | --- | ---: | --- |
| raw step33 | 0.010372 m / 0.052042 rad | 0.013141 m / 0.078789 rad | 否 | 否 | 1.161716 | 通过 |
| terminal058 step84 | **0.009064 m / 0.043876 rad** | 0.013067 m / 0.064304 rad | **是** | 否 | 1.524670 | 通过 |

首点原门为位置≤0.01 m、SO(3)≤0.05 rad，沿用`1e-10`容差。terminal058后移计划
的首点及原八周期box、动作变化率、关节缓冲、终端和0.1 s制动模型安全审核通过；
第二点位置与姿态均超原门，因此它不是双点或完整八点质量witness。后六点归一化缺口
分别以位置和姿态除以原阈值、减1后取最大值并截断为非负。

raw结果与此前原guarded路径的[后移检查](task2_shifted_seed_qualification_20261111.md)
一致，首点不合格；terminal058来自余量路径，与此前guarded路径的后移计划不同。
这只是对两个相关开发停止状态的冻结模型检查，不证明真实动作安全、持续可行或
跨初态泛化。若要使用terminal首点候选，须另行登记原完整安全审核、共享响应预算
和自身因果历史下的实际反馈与独立重放；不能追认为已派发。

## 复核与费用

两个计划均经独立标量响应重建，合计400个内环端点和5200模型转换步；最大关节差约
`4.44e-16`、TCP位姿差约`6.66e-16`。共使用2个静态模型、2次`mj_forward`、2次
八步rollout及2次独立审核；两个CPU worker与监督进程实际wait/reap、exit0、未发
信号，输入与源码前后一致。53项32并发CPU专项、全仓Ruff/格式和零模型dry通过。

本轮无新物理积分、新WAM、GPU或最终留出样本；`complete/passed`只表示检查闭合，
不授予控制或任务资格。原18正确参考、12开发及24最终留出门槛保持不变。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/task2_shared_shift_qualification_20261111/result.json` | `281b41e4a8170547f00fd3e381ade08a4643f88399d962602cdfe4cc966603fc` |
| `outputs/task2_shared_shift_qualification_20261111/workers.json` | `7a870ca449abcfb3a16757a0f64441c8fae25c33193b0fe678947bff1146a2b3` |
| `outputs/task2_shared_shift_qualification_dry_20261111/result.json` | `5e958ece9dc08d627ccd57cc30de0b0810d957d75782cb37a80e21672de03a07` |
| `outputs/qa/task2_shared_shift_qualification_20261111/result.json` | `3e8324d8f8e2d950a5b2b9bf7e10f9ba9f8c4949d6298c2e4d56cb8452ce3bf3` |
| `outputs/task2_shared_shift_qualification_supervisor_20261111/owner_exit.json` | `a74edc90cabb8049d14d6bf8294184b89b33c743648532c92baa3e5a713bfa74` |
