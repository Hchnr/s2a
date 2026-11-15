# 三任务因果力反馈：实际回放与按任务留出

2026-11-13。按[计划](../../../plans/v0.0.1/tracking/response_models/wrench_multitask_replay_20261113.md)
从 reset 回放三任务各两个 seed 的六条旧开发轨迹，分别执行 133、25、156、45、
462、204 个原动作，共 1,025 个 20 Hz 控制周期及 25,625 个 2 ms 内步。
全部 1,031 个端点在下一动作前读取原生当前 wrench；每条动作、状态、TCP 误差、
内环及夹爪反馈与原 producer 最大差均为 0，请求图像与任务标签一致。六个进程
正常退出并由调用者实际等待回收，共 18 个仿真模型及 18 个 OSMesa context，
没有新增 WAM 请求、动作选择或最终留出初态。原五条 owner 结束帧失败保持不变。

原登记审核器和两份测试后来修复，首次六个预检因此在启动物理前失败。本次从
Git 校验三份原始 blob 的 SHA，同时固定修复后的三个当前 SHA；修复审核器与
既有六案例独立审计所用版本逐字节相同，其他登记源原样核对。修正后的六个
预检、32 并发配置的 23 项回归和因果预测的 8 项回归均通过，无跳过或收集错误。
最终 64 并发配置的相关回归实际运行 32 项、32 个 worker，全部通过并回收；
全仓 Ruff 检查与 3,630 个 Python 文件格式检查通过。

## 跨任务的已执行动作预测

每条轨迹独立清空历史；排除六个没有上一周期历史的首动作，保留 **1,019** 个
周期。将同一任务的两个 seed 一起留出，仅用另外两个任务拟合；轮换三次。线性
模型输入为当前端点的世界系 wrench 和上一周期已完成的位置偏差，训练折内部
标准化、固定 ridge=1，旋转沿用上一周期的 SO(3) 偏差。`history linear` 使用
同样拟合方式但不含 wrench。原始模型与上一周期偏差是无需拟合的基线。
计划中采集段的“不新增训练”仅指重放采集；下述事后分析实际进行了 6 次拟合，
独立复核另进行了 6 次求解。

| 预测方法 | 位置预测 RMSE | 假达门 | 假停止 |
| --- | ---: | ---: | ---: |
| 原模型 | 1.088 mm | 18 | 0 |
| 上一周期偏差 | **0.493 mm** | **1** | 2 |
| 仅历史偏差线性拟合 | 0.842 mm | 5 | 2 |
| wrench + 历史偏差线性拟合 | 1.704 mm | 9 | 5 |

假达门表示预测的位置与旋转均在 0.01 m / 0.05 rad 门内，实际至少一项超限；
假停止则相反。保持这些原有门槛和全部案例分母，没有按结果筛选阶段或调参。
结果表明该简单 wrench 条件模型在本轮按任务留出的预测中明显退化，**不接入控制**。
cream/seed43 单例中上一周期偏差为 1 次假达门、0.669 mm RMSE，wrench 条件
模型为 9 次假达门、1.774 mm RMSE；cream/seed42 中后者有 3 次假停止。

这是对原动作的事后评分，实际改变动作会改变后续状态、参考和传感器读数；
它不是闭环对照，也不能把 1/1,019 当作未来安全概率。三个任务各只有一个旧
初态、两个 seed 相关，同一轨迹周期也相关；不能把周期视作独立泛化样本。
物体状态、接触标签和任务谓词均未作为模型特征，任务身份只用于划分训练与
验证；没有声称该结果能推广到新机器人、真机或最终保留初态。

一个事后分布诊断发现：留出 cream 时，664 个待评分端点中有 179 个的六维世界系
wrench 至少一个分量超出另外两个任务训练端点的逐分量最小/最大范围；该矩形
范围不是完整分布或因果证明。cream/seed43 的当前力范数最大仅约 5.95 N，
却有多次预测误差和实际 pose 超限，说明简单的大力阈值也不能覆盖这批失败。
下一步不继续扩大同类线性拟合，应优先追查各任务的参考质量与可执行性，并在
需要时研究有支持域约束的反馈，仍须实际控制和正确参考正控。

独立脚本直接从原 producer 和传感器轨迹重建训练/留出数据，用增广最小二乘
重新求解六次；参数及旧基线最大差 `3.70e-17`，留出预测最大差 `4.38e-16`。
其不重算原策略选择或质量停止原因，也不授予 18 条正确参考、12 条 WAM 开发
或最终 24 条验收资格。

| 证据 | SHA-256 |
| --- | --- |
| `outputs/wrench_multitask_replay_20261113/case_00/wrench_trace.json` | `b25606406148f6e577fedd804ba0cbeb7bd5f9592bb9a5316f55a659d145ec65` |
| `outputs/wrench_multitask_replay_20261113/case_02/wrench_trace.json` | `4110ea350c8503b540974c7071f73066427c044d4b8a86324ab46246557aad33` |
| `outputs/wrench_multitask_replay_20261113/case_04/wrench_trace.json` | `1ba6737c8a06b151d0e5de22edf75e11eb9eed6a59693296ab7e4e07d10f5e45` |
| `outputs/wrench_multitask_replay_20261113/case_06/wrench_trace.json` | `50e403f890938a6127ced3010e8fdc4d57f0a49ba02a24d6ea7372aeacaaf63a` |
| `outputs/wrench_multitask_replay_20261113/case_08/wrench_trace.json` | `f226133b6c674f95ef3f8845abff5fae3eabaa06cf1a7940be6a25c9755d35f6` |
| `outputs/wrench_multitask_replay_20261113/case_10/wrench_trace.json` | `45d827b5503dc59bac911944d622ece84b21cd1c230ad3d65743be95dc355fce` |
| `outputs/wrench_multitask_replay_20261113/cross_validation.json` | `cc5b29897300d5dc675798e99ce6c5a37b863f37ddac46591e60ac18d09fe51e` |
| `outputs/wrench_multitask_replay_20261113/wrench_coverage.json` | `e72f219a931451134c521a9498d5697513e907c178f378dc30d119006b879879` |
| `outputs/wrench_multitask_replay_20261113/independent_verification.json` | `6eecc6b5cb2fa970314a5f8a65b859713c28294306874fd0509897e2c1246715` |
| `outputs/qa/wrench_multitask_replay_v2_20261113/result.json` | `dca6a9b3f8af387067b7cdefc86e7374fb83270f1aa5ff52589fdda8a1be31b4` |
| `outputs/qa/multitask_wrench_prediction_20261113/result.json` | `92943e17e7e8696f52f280a2422b9c13af498f37ce36095ba1f14c15716dd1b8` |
| `outputs/qa/wrench_multitask_final_20261113/result.json` | `d84362e8ec4f48e60800ef34b4bf431cd958837cbcf9e148cf98e7e04590b410` |
