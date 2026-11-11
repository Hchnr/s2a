# 复位分段策略的首窗口实际等价

2026-11-11。按[预登记](../../../plans/v0.0.1/tracking/joint_control/segmented_reset_equivalence_20261111.md)，
对 12 条已审核 v4 开发案例从各自原状态和首图复位，构造带分段 owner/view 的统一
手臂与夹爪策略，分别执行已保存首个 WAM H40 参考的前 8 个控制周期。12 条来自
三个旧开发初态、两 seed 和两个参考变体；本次无新 WAM 请求、GPU、训练或最终留出。

## 结果

新 producer 12/12 完成 8 周期，独立 checker 12/12 重放完成，全部实际进程 exit0、
wait/reap、无信号。与[此前首窗口 producer](reset_start_feedback_20261111.md)逐项
比较，96 条命令逐分量完全一致；96 个完整实际状态、2,400 个 2 ms 内环时间/q/qvel、
TCP、开度/qpos/qvel/motor 以及最小关节余量的最大绝对差均为 **0**，endpoint receipt、
位姿门、grasp 和任务标签一致。本次 producer 与 checker 的同项最大绝对差也均为
**0**；checker 的 12 个 worker 均未加载候选策略。

新结果和原结果均有 96/96 位姿端点满足 0.01 m / 0.05 rad，全部实际内环余量
≥0.05 rad，最小 0.568645 rad，grasp 与任务成功标签均为 0/96。每步选择前检查原
命令，不一致则在派发前失败；实际完成后另核对状态与内环。原结果只用于对比与拒绝
不一致，不输入候选策略。初始创新/内环误差仍标明历史未知，未引入原轨迹未来状态。

分段历史的软件测试还覆盖已完成误差跨段保留、旧 view 失效和未完成事务拒绝换段；
这些是合成测试。**本次物理运行只执行首段，没有生成或接入第二个新图像参考**。
因此它验证在已测开发起点首窗口的数值等价，不能证明跨窗口物理行为、整任务成功、
新初态泛化、实时或真机效果。

## 资源与验证

本次 producer 和 checker 各用 36 模型/context、96 控制周期/2,400 内步，共
72 模型/context、192 控制周期/4,800 内步。producer 选择 96 条命令、调用质量
求解 96 次/实际 229 次响应，夹爪规划 96 次/629,856 个序列，witness 端点查询
19,200 次/模型转换 249,600 步；checker 未选择候选。只读总审计不创建模型或推进
物理。与此前现场准入 72 模型/context、原首窗口生产及重放 72 模型/context 合计，
这三阶段累计 **216 模型/context、384 控制周期/9,600 内步**；原准入的首轮失败
及其成本仍保留。

物理运行前 32 并发专项 69 项通过；只读审计 32 并发专项 34 项通过，均无跳过、
xfail 或收集错误。完整 pytest 尚未重跑；此前仿真环境全量 4 项失败、2 项跳过和
5 个文件收集错误仍保留。下一步需对固定开发案例登记新的当前图像 WAM 请求与
物理预算，实测首段完成后的参考切换和持续反馈；原 18/12/24 与最终稳定门不变。

| 仓库内证据 | SHA256 |
| --- | --- |
| [运行前 32 并发专项](../../../outputs/qa/segmented_reset_equivalence_20261111/result.json) | `629587f5a9c6c5f96107b9c02e6be1acdf8b29d4b50ad4887cef9602ca57378a` |
| [零模型 dry](../../../outputs/segmented_reset_equivalence_dry_20261111/result.json) | `c0d3263455a4acecb7cdf9bcc28199b5d9c932f8dc505c24242deb80edf934a9` |
| [producer](../../../outputs/segmented_reset_equivalence_20261111/producer/result.json) | `182f91a7603c94b58a9c0bd832773cbe8346a909517c4bda178b8ccf6c68d150` |
| [producer worker 生命周期](../../../outputs/segmented_reset_equivalence_20261111/producer/workers.json) | `60f70083fd2c81290a8eff48649a928e78baab47f69bd4ac32e9137cb5aa1ba6` |
| [checker](../../../outputs/segmented_reset_equivalence_20261111/checker/result.json) | `459656c0dff3cf43bfd16480316dcfbec0869da98443bf69a551ebf3c7a0b207` |
| [checker worker 生命周期](../../../outputs/segmented_reset_equivalence_20261111/checker/workers.json) | `11fdc9325ab1b6e16402785388f2f92f84860595e495fe4430d0568bcdc945c4` |
| [审计 32 并发专项](../../../outputs/qa/segmented_reset_equivalence_audit_20261111/result.json) | `1ca6d80359547403c123c74ea077157fda1cf9cba30c729d949c531aa9be7108` |
| [三方只读审计](../../../outputs/segmented_reset_equivalence_audit_20261111/result.json) | `d5f5da6635b6c2c297f38614b490fd539abec6ac45865079087b71935eb42066` |

执行和审计 QA 绑定的更新前 `docs/status.md` SHA256 为
`3340f55dccb8baa0666c356a7e84d54484a9dd064ad0aa160f0c71ea778547b8`；
原字节见[历史副本](../../../docs/history/experiments/status_before_segmented_reset_ee1afab5.md.txt)。
