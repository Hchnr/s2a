# 桌面参考投影：已记录状态的静态对照

2026-11-11。依据[预先固定的方案](../../../plans/v0.0.1/tracking/response_models/tabletop_reference_projection_20261111.md)，
对 correct_08 的 step80–126 共 47 个已记录状态，使用各步当时的夹爪配置、
已生成 TCP 目标和离线模型中的桌面盒体，比较原参考与两种投影。step80–125
有已执行动作；step126 的目标已生成但动作未派发。每步的状态取自原轨迹，
不是投影候选产生的反事实状态。

投影只沿桌面法向抬升 TCP，保持目标旋转，并以当前 finger mesh 顶点构造
条件包络；它不改变夹爪命令。零位姿误差候选使用 `0 m / 0 rad`，位姿包络
候选使用 `0.01 m / 0.05 rad`；二者均假设当前夹爪形状不变，净空为 `0.001 m`，
最大参考平移为 `0.05 m`。超过上限将显式拒绝，不产生投影目标。

| 静态指标 | 零位姿误差候选 | 位姿包络候选 |
| --- | ---: | ---: |
| 保持原目标 | 40/47 | 38/47 |
| 抬升目标 | 7/47（step120–126） | 9/47（step118–126） |
| 超上限拒绝 | 0/47 | 0/47 |
| 抬升量最大值 | 0.030104 m | 0.042691 m |
| 仅在抬升状态上计算的中位抬升量 | 0.017034 m | 0.024361 m |
| 投影后的 finger2–桌面静态接触 | 0/47 | 0/47 |

原始目标的静态放置在 step120–126 均与桌面接触；候选逐点抬升量如下。
当前状态相对原参考的位置误差基于原轨迹测量，不是投影控制后的误差。

| step | 已执行原动作 | 原目标静态接触 | 当前状态相对原目标位置误差 | 零误差候选抬升 | 位姿包络候选抬升 |
| --- | --- | --- | ---: | ---: | ---: |
| 118 | 是 | 否 | 0.006544 m | 0 | 0.001350 m |
| 119 | 是 | 否 | 0.007255 m | 0 | 0.008733 m |
| 120 | 是 | 是 | 0.005950 m | 0.001366 m | 0.014148 m |
| 121 | 是 | 是 | 0.005971 m | 0.006463 m | 0.019234 m |
| 122 | 是 | 是 | 0.012860 m | 0.011737 m | 0.024361 m |
| 123 | 是 | 是 | 0.019212 m | 0.017034 m | 0.029521 m |
| 124 | 是 | 是 | 0.024251 m | 0.021976 m | 0.034352 m |
| 125 | 是 | 是 | 0.026969 m | 0.026442 m | 0.038855 m |
| 126 | 否 | 是 | 0.028369 m | 0.030104 m | 0.042691 m |

位姿包络候选从 step120 起的抬升已经超过原参考的 `0.01 m` 位置容差；
因此，即使未来控制器能精确到达该投影目标，也不能把它记作原参考跟踪达标。
本次结果只支持将候选送入进一步研究，不证明机器人能到达这些姿态或完成任务。

独立复核不调用投影函数：从原始状态使用另一个模型重建当前 TCP、桌面与全部
18 个编译 finger mesh 顶点，逐顶点用标量计算误差包络和法向抬升，并核对原始
和投影参考误差。47 行、两种候选的判断一致，TCP 局部顶点最大差
`3.47e-16 m`。首轮使用 1 个模型和 188 次 `mj_forward`，复核使用 1 个模型和
47 次 `mj_forward`；均无物理积分、WAM/GPU 调用或新初态，两次 worker 均实际
wait/reap。对应 32 并发专项分别 17 和 18 项通过，无跳过或收集错误。

最终 64 并发投影及相关几何专项 **31 项通过**。模拟器环境的 64 并发全仓
回归收集到 6,677 项，其中 6,671 项通过、4 项失败、2 项跳过，另外有 5 个
文件因该环境 NumPy 1.26.4 与 `pyarrow` 不兼容而收集失败；4 项失败均是未修改的
clean streaming 加权最小二乘测试，观察到约 `4e-14` 的数值差异。改用仓库
`.venv` 的 NumPy 2.1.0 / `pyarrow` 23.0.1 后，那 5 个文件的 14 项及 clean
streaming 文件的 26 项分别通过。环境切换复核不能抹去模拟器环境的全仓失败。

边界：桌面盒体来自仿真模型，只供离线诊断；在线部署需要独立获得可靠标定。
固定形状误差为零不能代表夹爪执行时不会运动。条件净空只覆盖指定 finger mesh
与这个桌面盒体，不覆盖其他机器人部件、物体、路径或碰撞动力学。尚未在真实
反馈历史下执行候选，也未验证任务成功、独立新初态或真机；原 18/12/24、每条
600 周期和稳定成功门槛均未改变。下一阶段须先登记有限的物理反馈对照、夹爪
形变条件和拒绝策略，使用相同控制接口并同时评价原目标与投影目标。

| 仓库内证据 | SHA256 |
| --- | --- |
| [首轮零工作 dry](../../../outputs/tabletop_reference_projection_dry_20261111/result.json) | `3edbb3d780de250fc53e32a76bc4391085f91014e133d8f756444fa85a2f9c9b` |
| [首轮 32 并发专项](../../../outputs/qa/tabletop_reference_projection_20261111/result.json) | `07756cd81875c990e29ad938c7141509b347598edf219fa8a7fe3903a008b14e` |
| [首轮 owner](../../../outputs/tabletop_reference_projection_20261111/result.json) | `fdc67c5f9c0bdffe3941422fa30ca302747055e1f8f61ce80d29eeecdfb91da5` |
| [逐状态结果](../../../outputs/tabletop_reference_projection_20261111/worker/result.json) | `c6886b32040c75f3591599ec3ed3ae44a2f6d0be2cb32077f781ab0dc9bf122a` |
| [复核零工作 dry](../../../outputs/tabletop_reference_projection_audit_dry_20261111/result.json) | `1a9c668de5452dcabc640aedb8a4a8fcef80af0a648b39b64e79139819f22cbe` |
| [复核 32 并发专项](../../../outputs/qa/tabletop_reference_projection_audit_20261111/result.json) | `41327435a087d9da65bcbfd6918b6150ad576478d83be9eaf6c282f822d47f20` |
| [独立复核 owner](../../../outputs/tabletop_reference_projection_audit_20261111/result.json) | `1fd9fc5bafbac057a67c9cb0d63704bb27394ec5ed74eadd9b8ba5e5ecee25f0` |
| [独立逐状态复核](../../../outputs/tabletop_reference_projection_audit_20261111/worker/result.json) | `500f4bd3abbd3d358573e2615e9f75a9ffd0c7c31a7d51955c4c660af1cd05a2` |
| [最终 64 并发专项](../../../outputs/qa/tabletop_reference_projection_final_20261111/result.json) | `08b4413d54c7d848c1dce7a986bfd11ea7f25c5605de2e1ba58001cf8b1d9053` |
| [模拟器环境全仓失败记录](../../../outputs/qa/tabletop_reference_projection_full_20261111/result.json) | `e300b0254116144f5e6afc011c5b3a1869ca96c96859921c7c2749c36ae6ced8` |
| [兼容环境数据文件复核](../../../outputs/qa/tabletop_reference_projection_data_20261111/result.json) | `c8ff7e3ad522b7d85adce126728cb394dcc00a06a704fcbdfc85498e7968dd06` |
| [兼容环境 clean streaming 复核](../../../outputs/qa/tabletop_reference_projection_clean_streaming_20261111/result.json) | `f3f90c55ec796b7fe92b81dd435e650a40c12fa20011dc41879978d559bc807f` |
