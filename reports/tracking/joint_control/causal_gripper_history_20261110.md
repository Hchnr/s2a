# 两处闭合机会的因果夹爪历史重建

2026-11-10。根据[登记方案](../../../plans/v0.0.1/tracking/joint_control/causal_gripper_history_20261110.md)，
从两条已物理重放的seed42开发前缀重建原 `LoadedVelocityControl`。每一步只使用
当时的实测、已完成动作和已生成参考；对象、接触和任务标签没有作为控制输入。
重建器与历史控制器使用相同实现，因此本结果是日志一致性证据，不是独立算法证明。

| 开发案例 | 核对周期 | 规划调用 | 完整候选序列 | 最大记录差 | 起点intent方向 | 起点闭合状态 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| task0 s42 raw，终点136 | 136 | 136 | 892296 | 0 | 闭合方向 | 未进入闭合保持 |
| task1 s42 terminal058，终点360 | 360 | 360 | 2361960 | 0 | 打开方向 | 未进入闭合保持 |

496次重建全部选择历史 `model_tracking` 路径；预览、夹爪滞回、规划序列、响应和
最终夹爪命令与原日志逐项一致，最大数值差为0。两个终点的前一夹爪命令均为0，
但intent方向分别为 `+1` 和 `-1`，原参考极值分别为 `0.6562155485` 和
`0.9086179733`；不能仅凭相同的前一命令重置或推断相同策略状态。完整状态、
历史与来源保存在逐案例结果中。

实际共496次有限planner调用和3254256条完整候选序列；两worker均实际wait/reap、
exit0且未发信号，源码及输入前后一致。115项32并发CPU专项、零模型dry及全仓
Ruff/2743文件格式检查通过。本阶段调用了夹爪数学响应模型，没有新仿真模型、
物理步、世界模型推理、GPU或final fresh。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/causal_gripper_history_20261110/result.json` | `7e919a511ab13dedba478d521cb52e13a2294b6364e46b0b76f049567abf168b` |
| `outputs/causal_gripper_history_20261110/case_00/result.json` | `c0327716949f567c670c7112adc0a65179f981e970072f7f19cad377a6e93281` |
| `outputs/causal_gripper_history_20261110/case_05/result.json` | `cfab7fb0f54c346dde3ff59d6469e70f3dc93a2f28b9ce99b5dc1ae2b19bce85` |
| `outputs/causal_gripper_history_dry_20261110/result.json` | `956d8b4f6bf02bb340e9f98c540e6ba725e51a688a49154f127483c48e9ec35f` |
| `outputs/qa/causal_gripper_history_20261110/result.json` | `5130496776ae134fd025d74dc8254c9c3798352cf46d0b96a799b16d9ef08a4c` |

下一步应在每个匹配物理起点分别恢复该夹爪状态与手臂完成历史，比较当时已经生成
的旧/新参考；按原关节缓冲和控制接口验证实际动作及独立重放。这里没有执行新
参考选择或测试抓取成功，原18正确参考、12开发和24最终留出门槛保持。
