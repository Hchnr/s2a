# DROID episode 0 — 2026-10-16

证据范围：只读日志、几何/参考接口代数、单 episode 辨识模型内 rollout。没有独立物理仿真或真机验证。

代码 `b82b0b619dcc8725182aecf6b6d59d065eb27eb1`；dirty=`False`；seed=42。
sample SHA256 `06ae64d110f9dec36975a55806b27387a994c36d451258cac1f136273dfce388`。完整文件/代码指纹与配置见 sample_manifest.json。

## 数据与几何

选定 300 个转移，保留 304 个观测（含历史与 endpoint）。
物理时间 [2.106992219, 23.339967628] s；媒体时间 [2.0, 22.0] s。

| 检查 | 位置最大误差 m | 旋转最大误差 rad |
| --- | ---: | ---: |
| single_step | 5.71046e-08 | 5.52152e-08 |
| integrated | 4.16351e-08 | 3.76671e-08 |
| tcp_to_r1_flange | 3.03975e-08 | 3.04117e-08 |
| tcp_flange_roundtrip | 3.10317e-17 | 0 |

observed 下一状态相对原 commanded target 的 RMSE：0.0213747 m / 0.0557798 rad。
公开参考接口解码历史命令的 target RMSE：0.00518253 m / 0.0134473 rad；历史接口匹配未确认。

## 留出响应与命令诊断

模型 `diagonal`，history=3，lag=0，ridge=0.0001；fit samples=177。
计划命令时域 0.070980216 s（只取 fit median）；反馈时域使用已发生的物理时间差。

| 区间 | 一步 NRMSE | 上一运动 NRMSE | 多步 NRMSE | 多步位置 RMSE m | 多步姿态 RMSE rad |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.130463 | 0.220406 | 0.690299 | 0.053655 | 0.192583 |
| test | 0.089895 | 0.122318 | 0.771023 | 0.101210 | 0.344457 |

每个独立区间清空命令/运动历史与延迟队列，剔除 8 个 warm-up 转移；validation/test 计分分别 32/52 转移。
归一化尺度来自 fit。lag 用真实物理毫秒数另存 JSON。接触标签缺失，不虚构接触/自由运动分段。

| 命令诊断 NRMSE | hold/zero | previous command | A | linear calibrated |
| --- | ---: | ---: | ---: | ---: |
| fit | 1.139235 | 0.165728 | 0.954786 | 1.552134 |
| validation | 2.085350 | 0.165128 | 1.627156 | 1.766943 |
| test | 1.614890 | 0.080250 | 1.405228 | 1.125336 |

命令诊断共有支持剔除 history/lag 边界，fit/validation/test 为 172/32/52；不以 command RMSE 选择 PID。
线性逆是正则无约束解后范数投影；模型 history 中的运动量只使用已发生的观测。

## 模型内控制对照

增益只在 validation 选择；FF/P/PI/PD/PID 具有相同前馈与限制，所有场景/候选/划分分别初始化。
若 validation 全部候选含无效场景，仅报告网格第一组固定诊断，不将其视为选定/整定成功（selection_status 另存）。
初偏、gain=.7、额外一帧延迟、恒扰动、噪声、静止目标均预先固定。参考是 oracle；每一步状态由候选命令推进。

| 前馈 | 方法 | test 场景平均 tracking score | 场景全部有效 | 全扰动验收 |
| --- | --- | ---: | --- | --- |
| analytic | FF | 9.541935 | True | baseline |
| analytic | P | 4.511901 | True | False |
| analytic | PI | 4.311155 | True | False |
| analytic | PD | 4.512345 | True | False |
| analytic | PID | 4.311662 | True | False |
| linear | FF | 40.253341 | True | baseline |
| linear | P | 39.329239 | True | False |
| linear | PI | 39.324052 | True | False |
| linear | PD | 39.326227 | True | False |
| linear | PID | 39.320244 | True | False |

tracking score 按 .01m/.05rad 尺度平衡（越小越好），不等同任务成功率。
完整逐场景位置/姿态 RMSE、终点、静止稳态/超调/调节时间、命令变化率、饱和比例/时长与验收细项保存在 closed_loop_model_metrics.json。
任何全扰动验收失败的方法均不能宣称达到方案的扩展标准。模型自洽也不构成真机稳定性证据。

| 未执行项 | 缺失条件 |
| --- | --- |
| gain_scheduling | P4 requires verified same-device multiple episodes |
| direct_inverse_MLP | P4 requires episode-level independent splits |
| residual_MLP | P4 requires episode-level independent splits |
| independent_simulator | no verified matching controller/robot configuration |
| hardware | robot serial, controller config and execution channel unverified |
| world_model_reference | frozen predicted chunks not provided |

## 产物

sample.npz、sample_manifest.json、data_audit.json、analytic_commands.npz、geometry_metrics.json、adapter_params.json、pid_config.json、calibration_metrics.json、forward_replay.npz、closed_loop_model_metrics.json、closed_loop_traces.npz。
源数据保持只读，完整产物在 outputs/droid_episode0/，不进入 Git。精简结论及摘要在 reports/ 与 plans/v0.0.1/。

![forward_replay.png](../outputs/droid_episode0/figures/forward_replay.png)

![tracking_comparison.png](../outputs/droid_episode0/figures/tracking_comparison.png)

![analytic_initial_offset.png](../outputs/droid_episode0/figures/analytic_initial_offset.png)

![linear_initial_offset.png](../outputs/droid_episode0/figures/linear_initial_offset.png)

