# terminal058 step85的首点优先搜索

2026-11-11。按[预登记](../../../plans/v0.0.1/tracking/joint_control/task2_terminal85_first_search_20261111.md)，
在[独立重放的回退停止状态](task2_causal_shift_fallback_20261111.md)terminal058
step85，以已保存LP初值和已完成计划后移初值，分别运行冻结的两阶段首点优先搜索。
两种种子的共同关节仿射安全域最大差约`1.78e-15`，终端方向为空。

| 初值 | 响应 | 已评估线性安全计划 | 首点候选 | 最近安全尝试：位置 / SO(3) | 后七点最坏归一化缺口 |
| --- | ---: | ---: | ---: | --- | ---: |
| 已保存LP初值 | 320 | 11 | 0 | 0.010790 m / 0.057714 rad | 2.149896 |
| 后移已完成计划 | 320 | 20 | 0 | 0.010813 m / 0.057655 rad | 3.174617 |

两次第一阶段均报告优化器成功终止，但首点超限平方目标仍约0.015018；第二阶段
均用尽320响应预算，实际迭代数未知。首点原阈值为0.01 m / 0.05 rad，沿用
`1e-10`审核容差；两次均无首点候选，因此没有独立候选审核或动作派发。
最近安全尝试只在各自有限缓存内选择，不是全局最优或可派发命令。

先前[首点外界](task2_fallback_stop_pose_bound_20261111.md)未排除该状态的模型
可行性；本次失败仍不能证明冻结模型或真实物理不可达。保存LP初值只重建了原反馈
停止时的一个可行起点，不是原始基线细化种子或原停止求解的精确重放。不能用
首点搜索失败或较小局部误差替代完整任务评价。

## 费用与复核

实际使用1个静态模型、1次`mj_forward`、2次准备rollout、2次搜索共640响应，
两初值均已安全可行，本次未调用初值LP。没有合格候选，独立候选审核0次；1个
worker和监督进程实际wait/reap、exit0、未发信号，输入和源码前后一致。52项
32并发CPU专项、全仓Ruff/格式和零模型dry通过。本轮没有新物理积分、新WAM、GPU
或最终留出样本；`complete/passed`只表示对照按预算完成。

该结果支持继续研究更早的因果状态/参考衔接或不同的有界搜索，而不是把双点搜索
的缓存或两种首点起点视为充分覆盖；原18正确参考、12开发和24最终留出门槛不变。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/task2_terminal85_first_search_20261111/result.json` | `ff6562c126a91c6104ab6378fbac2929bdaed162d851d6d890c124d39dd502be` |
| `outputs/task2_terminal85_first_search_20261111/workers.json` | `68083b46fce508f75ac9522d742ae40d78d1e5d92dab913a98c7dddb2fad181f` |
| `outputs/task2_terminal85_first_search_dry_20261111/result.json` | `9b3d1ec656636268daf4731fda29a7b127109fedb674b39fbc28149042f7e575` |
| `outputs/qa/task2_terminal85_first_search_20261111/result.json` | `825ed1773a4486ad31bbde0df083e1e571e82fa2dfc4475e7b8d3ebc6b6a7b3d` |
| `outputs/task2_terminal85_first_search_supervisor_20261111/owner_exit.json` | `10718fc7f866dd61303e0964d25bb33dc39b1bd14bb189f2e56b0ab1fee6a552` |
