# 已保存 WAM 参考上的数值修正物理反馈

2026-11-11，按[预登记](../../../plans/v0.0.1/tracking/joint_control/quality_rate_repair_recorded_feedback_20261111.md)，
在 correct_08、seed 42 上重建 80 周期正确参考前缀，再使用先前真实 WAM 生成的
六份已保存响应。每份响应仅在新运行的当前 step、时钟、原生状态 SHA 和相机图像
SHA 与原请求一致时复用；本次没有新 WAM/GPU 推理。可注入求解器保留原质量门、
关节缓冲与独立 witness 审核，默认在线路径未更改。

producer 前 **45** 个反馈周期与[原真实 WAM 运行](../references/causal_stream_live_cached_20261111.md)
的命令、完整状态、内环关节/速度和时间最大差均为 **0**。原运行在 step125 停止；
本次求解器生成并独立审核修正候选，实际多执行 **1** 周期，在 step126 因再次
没有合格质量 witness 于派发前停止。共完成 46/48 个反馈周期，末尾任务谓词连续
为真 **39 个端点**，仍未达到原 40 个连续端点门。整个续段位姿门 42/46，第六
窗口为 2/6；最小关节余量约 0.765624 rad，无小于 0.05 rad 缓冲或越界内环样本。

| step125 首周期误差 | 冻结模型预测 | 实际物理 |
| --- | ---: | ---: |
| TCP 位置 | 0.010000000000164 m | 0.0248504 m |
| TCP 旋转 | 0.0321321 rad | 0.0811767 rad |

模型位置仅在原 `1e-10` m 数值容差内达门；实际位置和旋转均超原 0.01 m /
0.05 rad 门。该周期的任务谓词仍为真，但局部模型 witness 并未保证实际跟踪。
step126 的停止原因由 producer 记录，独立 checker 不重算求解器，不能把重放
描述为对无解原因的证明。

独立 checker 没有加载修正求解器，重放全部 46 条动作及 1,150 个续段 2 ms 内环
样本；状态、关节位置、速度和时间最大差均为 **0**，任务、抓取和位姿门标签
一致。producer 与 checker 各完成 126 控制周期、3,150 MuJoCo 内步，所有拥有的
进程均正常退出并实际 wait/reap。producer 共使用 3,906 次质量模型响应和
9,200 端点查询 / 119,600 模型转换步的独立核验预算；六份响应的 210 次去噪
属于历史运行，新增模型初始化、请求和去噪均为 0。

运行前 64 并发相关专项 132 项通过、零跳过和收集错误；另有 32 并发来源审计
专项 12 项通过。本次是单一已研究初态、暂停推理延迟的开发物理反馈，不是
600 周期完整任务、实时控制、18/12/24 留出验收或真机成功。下一步应研究局部
模型预测与实际响应的偏差及质量裕量，同时保留原安全和停止门。

| 仓库内证据 | SHA256 |
| --- | --- |
| [零物理 dry](../../../outputs/quality_rate_repair_recorded_feedback_dry_20261111/result.json) | `1c6b2fce85a5456299cd76c038ee44342b7397e049eec19141082c7e7845f791` |
| [64 并发运行前专项](../../../outputs/qa/quality_rate_repair_recorded_feedback_20261111/result.json) | `2136e8e3f6df23cc1a7c25f9bef895c5f3ee0ae36201a1a5a57578f1d34a697b` |
| [producer owner](../../../outputs/quality_rate_repair_recorded_feedback_20261111/producer/result.json) | `2c3ab4ea6601fa38bc3db010dc26d3376d7a87ce90fd6098be1f685df512596a` |
| [producer 物理结果](../../../outputs/quality_rate_repair_recorded_feedback_20261111/producer/simulation/result.json) | `e936271d98ba46fc975c2eea3e2808930f6d7d895c0c4e18db254e08b1be154d` |
| [历史响应 relay](../../../outputs/quality_rate_repair_recorded_feedback_20261111/producer/model/result.json) | `d1d1c441d220ebc187859ed9345a55ad7ead6cf51d68e7fba5ae41f535be3042` |
| [checker owner](../../../outputs/quality_rate_repair_recorded_feedback_20261111/checker/result.json) | `bef1e291dd5c8735c61656c475ea15fb0287aeb91c6509b913a409e8541f55fa` |
| [checker 物理结果](../../../outputs/quality_rate_repair_recorded_feedback_20261111/checker/simulation/result.json) | `de27191cf2001e00b4b2c3cd2abd6e10488ebeb02f950f4435d535f5db3ea5f0` |
| [只读来源与重放审计](../../../outputs/quality_rate_repair_recorded_feedback_audit_20261111/result.json) | `bd33ed32be8e4bb4a9ef0c3fb753a8d0883e7cb6bd3ef22fedd9729807b4c802` |
| [32 并发审计专项](../../../outputs/qa/quality_rate_repair_recorded_feedback_audit_20261111/result.json) | `4bbc41b36fec2535301cdebbb9756f4cbb178c4698f9e77b205efd2f1ff88ccf` |
