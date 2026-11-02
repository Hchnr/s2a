# pose下限对齐的实现与完整新方向验证

2026-11-02，实现/预登记996a90a，按8358769设计。对齐后的新方向有安全且reserve
合格的下降步；尚无完整跟踪witness，不能替代连续迭代、物理或任务成功。

## 实现和独立检查

ReserveAlignedCapObjective显式使用cap_distance_gn_with_reserve_entry_floor策略。
pose阶段floor=max(original floor, buffer+entry fraction*target/weights)，仅修改
原safety行b，A/B/g、其他约束行、原.01m/.05rad caps和实际接纳门保持；restore
问题逐值不变。原terminal time scale用于raw余量换算，当前可接纳pose anchor的
零步仍满足该下限。旧默认类和driver语义保持。

独立ReserveAlignedCapAuditPolicy自行计算原reserve及新floor，不调用生产helper；
默认审核拒绝新策略tag，必须显式选择独立策略。cap ledger审核及共享固定QP
runtime仅增加可选policy参数，旧默认路径保留。完整B/g/A/b在QP前逐值比较。

最终43项jobs32专项全部通过、0skip/collection错误，worker回收、来源不变。
覆盖pose仅收紧、其他矩阵/行不变、restore exact、terminal单位、零步可行、
较严原floor保留、独立性、完整ledger和损坏安全证书拒绝及旧默认等。
初测35项有1条fixture错误：测试假定证书保存b字段，实际仅保存诊断；修正为损坏
真实安全证书字段后完整最终轮通过，未更改生产算法或容差。首次失败记录保留。
Ruff/format及Git空白通过，未运行全仓回归。

另重放旧sum-q的55个保存QP、122完整响应，所有QP输入和完整答案完全一致，
SHA仍26c0fcd1286c08a503ca8899007416f5cefdab2e6fd0401a7da37402fa85a258。
该回放没有新增QP/NNLS/机器人响应，不能计为新实验成功。

## 全部预先固定尺度

固定anchor85、刷新Js/Jq/Je、原root域与pose ceiling，完整cap-distance目标。

| alpha | joint/buffer安全 | cap loss | 实际下降 | 下降比值 | reserve | 原reserve接纳 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| anchor | True | 2.0100673083 | 0 | — | .997067 | — |
| 1 | True | 2.0109296371 | -.0008623289 | -.555361 | 1.306534 | False |
| .5 | True | 2.0098796989 | .0001876094 | .220702 | 1.056328 | True |
| .25 | True | 2.0096831901 | .0003841181 | .866201 | .960298 | True |
| .125 | True | 2.0098435291 | .0002237791 | .988723 | .979182 | True |
| .0625 | True | 2.0099533061 | .0001140021 | .997244 | .988268 | True |
| repeat | True | 2.0100673083 | 0 | — | .997067 | — |

全部caps12/24、prefix2、witness False，pose ceiling均通过。alpha1虽安全但loss
变差，不接纳；按原顺序首个合格步为.5，不按最优观察loss挑选.25。
旧未对齐QP相同步长reserve=.854603，对齐后1.056328；支持该新方向上避免立即
restore，但不证明长期改善。上述接纳是固定探针后的原规则分类，未执行搜索更新。

实际1QP/26迭代/1NNLS，原1e-6证书、scaling/recovery通过。生产7响应/2100内步/
2185查询，独立7/2100/2185，共14/4200/4370、2model，0新模型拟合。
原1e-10/torque1e-8布尔审核、完整响应、历史anchor/repeat exact，源码/输入前后
SHA不变。0物理/GPU/WAM/fresh/最终留出/实际参考消费，候选未下发；不裁剪、
换方向、追加预算或重试。CPU空CUDA/LD、数值库线程1。
tool session18362退出0、实际wait/reap完成，PID221049确认不存在，600秒TERM/
10秒KILL上限未触发。

| 产物 | outputs相对路径 | SHA256 |
| --- | --- | --- |
| 完整结果 | aligned_cap_distance_probe_20261102/result.json | 92b368dc009dcf0ae77e56283fdae7f9a63bfb4ccc294ea092da3eba6681373a |
| QP证书 | aligned_cap_distance_probe_20261102/qp_result.json | 1b3273c00b83f1f857e714ba59d503bc7e4d4319289b309bc3874997a3f6d60e |
| 保存分析 | aligned_cap_distance_probe_20261102/saved_analysis.json | 05ccf595227f0dfda44e5e7fd38db9d4f37336a89907f651a4569d71e4126552 |
| owner终态 | aligned_cap_distance_probe_owner_20261102/observation.json | 65d4127c4572f5a497a46a4dfdd5dfea9d572c337cc41890ae719c1857e4762f |
| 最终43项QA | qa/aligned_cap_distance_probe_20261102/result.json | 793950c10c25a189e8c787a7ba45a6aad7e5c30cba12efaf778196c034f9e7f2 |
| 默认闭合等价 | reserve_aligned_cap_implementation_20261102/sumq_equivalence.json | 5a0b81ce90a06f9bed4ede65870f7f445003b303dbbcff043c44c4975e955e1d |

下一步显式登记同根域/模型的有限连续迭代，比较停滞、恢复、完整跟踪和费用，
不把单步对齐效果当作最终成功。实际reserve仍可能因非线性跌破门，必须按原规则
拒绝；总目标、原18/12/24及稳定20/24、每task4/8验收仍未完成，GPU禁用。
