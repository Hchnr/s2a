# 有限释放保持的匹配物理反馈

2026-11-11。按[预登记方案](../../../plans/v0.0.1/tracking/joint_control/bounded_release_hold_feedback_20261111.md)，
在已研究的 task1 seed42 terminal058 原初态重放 360 个原生命令，随后先执行已生成
旧参考 8 周期，再根据实际完成状态选择有限保持及之后的衔接预览。新候选只执行保持
阶段的 8 周期；衔接和合法释放均未执行。本次没有新 WAM 查询或最终留出实验。

| 分支，切换后的 8 个端点 | 对所选参考位姿达标 | 对最新参考位姿达标 | grasp 端点 | 末端物体高度变化 | 内环最小关节余量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 本次有限保持 + 后续衔接预览 | 8/8 | 0/8 | 8/8 | +0.050281 m | 0.643644 rad |
| [固定旧参考](causal_event_long_feedback_20261110.md) | 8/8 | 0/8 | 8/8 | +0.053149 m | 0.633064 rad |
| [立即衔接参考](causal_event_bridge_feedback_20261110.md) | 8/8 | 2/8 | 3/8 | −0.000013 m | 0.764479 rad |

位姿双阈值为实际 TCP 位置 ≤0.01 m 且 SO(3) ≤0.05 rad，未包含 opening。物体
高度相对本对照 step360 起点，grasp 是环境原生端点标签；不能推断整个周期的接触
或最终放置。三组在共同前缀后产生不同实际状态，均无任务成功，不是同一后续状态
的在线反事实。此次候选虽然保持 8 个接触端点并抬升约 5.03 cm，但与既有旧参考
分支的接触长度相同，不能声称新方法改善抓取或完成任务。

本次保持阶段相对所选旧参考的最大实际位置 / 旋转误差为
`0.002359 m / 0.030223 rad`；相对**最新**参考分别为
`0.049553 m / 0.409531 rad`，最新参考 opening 的 8 点 RMSE 为 `0.724247`
（归一化开度），末端实测 opening `0.239902`。固定旧参考和立即衔接的最新位姿
最大误差分别为 `0.051609 m / 0.425692 rad` 和 `0.022657 m / 0.103835 rad`。
保持参考对自身达标不能掩盖对最新目标的显著偏差；opening、夹爪 motor 命令与实际
测量分别保存在证据中。

前 8 个新周期的动作、完整状态、200 个内环样本、两套参考误差及标签均复现既有旧
分支。端点368使用刚完成周期的实际内环，最小余量 `0.802340 rad`，以及当前
`0.254870` opening 和已完成 closing 状态，选择 `bounded_hold`。手臂、夹爪与
已完成历史在参考重绑定时保持。后续 8 个保持周期全部实际完成，参考事务才提交；
提交后状态为 held，桥接未派发。由于候选的未来预览已改变，保持阶段首条动作即与
旧分支不同；不能将旧分支的未来轨迹冒充新候选结果。

本次 16 个新周期的 400 个 2 ms 实际内环样本均守住 0.05 rad 缓冲，0 越界，
16 个端点均无任务成功。保持阶段已派发决策约 `1.099–1.244 s`，超过 20 Hz 的
0.05 s；仿真等待时物理时钟暂停，不具实时资格。此单个已研究状态的短窗结果不
证明可安全释放、持续重规划或在其他任务 / 初态稳定控制。

## 独立复核与费用

checker 未加载候选选择器、手臂或夹爪候选策略，也没有重新求解决策；从原初态
重放全部 16 条实际命令后，完整状态、400 个内环 q/qvel/time、所选及最新参考误差、
opening / motor、物体位置和任务 / grasp 标签与生产记录最大差均为 0。前缀末周期
25 个内环样本也一致；参考选择及切换决策标记为未重新计算。

producer 和 checker 各执行 376 个控制周期 / 9400 个 2 ms 步，各初始化 3 个模型
和 3 个 OSMesa 软件 context；合计 752 周期 / 18800 步。producer 选择 16 条命令、
进行 16 次质量求解 / 1288 次模型响应，夹爪历史重建与提案共 361 次 planner /
2368521 条候选序列；checker 不新选动作或求解。两阶段 worker 和监督进程均实际
wait/reap、exit0、未发信号，输入与源码前后不变。执行源码提交为
`41c5a4be`；32 进程 CPU 专项 113 项通过、0 跳过，零模型 dry、全仓 Ruff 及格式
检查通过；未重跑完整测试集。`complete/passed` 表示执行和复核闭合，
`control_qualified=false`。

| 证据（相对 `outputs/`） | SHA256 |
| --- | --- |
| `bounded_release_hold_feedback_20261111/producer/result.json` | `04bf0a22a2a9a5867cd115965e6e475e78ec3ab64d233c4842388d54f0abe8b7` |
| `bounded_release_hold_feedback_20261111/producer/workers.json` | `6c6ac63ba6957d7dd45367475287d0cb9dd1aaddf407ef08b3946ef3d55db1ae` |
| `bounded_release_hold_feedback_20261111/checker/result.json` | `3829e82d72a2734460ff383aa71faf2505b9b4b3c60ed2393c062a6f396b4a9a` |
| `bounded_release_hold_feedback_20261111/checker/workers.json` | `6778b15f8c4767b740c63eafeac8b96445f364c7950e835a01f9ee1afa8e2bfe` |
| `bounded_release_hold_feedback_supervisor_20261111/producer_exit.json` | `579058b9795e83ea0b3e22ab95eee3866aa32a8a4668927e7f4e153f9aea8037` |
| `bounded_release_hold_feedback_supervisor_20261111/checker_exit.json` | `b8cd05bb0d4927f5ea533efe0a48cbf1bed030d523a85144a582060a58e6acce` |
| `bounded_release_hold_feedback_dry_20261111/result.json` | `9b94d4daf10f4d6c3044506185df58abbe513bd8ab309c006131ad453705ff38` |
| `qa/bounded_release_hold_feedback_20261111/result.json` | `e573641576805436cc4ebbad23f5c940fda933d61dc05206cb13f4461792be54` |

下一步须在本候选实际保持结束的状态上验证衔接和释放，并将最新参考的位姿 / 开度
误差与任务结果同时计入；不能仅延长旧参考或用接触端点代替目标。原 18 正确参考、
12 WAM 开发、24 最终留出及 ≥20/24、每任务 ≥4/8 门槛保持，GPU 继续禁用。
