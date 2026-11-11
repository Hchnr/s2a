# 桌面参考投影：短窗口物理反馈对照

2026-11-11。按[固定设计](../../../plans/v0.0.1/tracking/response_models/tabletop_projection_feedback_probe_20261111.md)，
在 correct_08、seed 42 的同一已审核状态从 step120 分叉。先重放 step0–119，
核对每个周期的状态和 25 个内步；三个分支在派发新动作前核对同一状态哈希、
控制器目标、上一命令和夹爪电机缓存。只使用分叉前 step120 图像已经生成的
参考，沿各分支自身的实际状态执行 step120–126；step127 前按预算停止。
step126 在原历史轨迹中未派发，但本次已在三个实验分支中实际执行。

各分支使用同一个单步 OSC 比例控制及接口、相同动作范围和变化率上限，七步内
保持分叉时的夹爪命令；只改变是否对当前目标做桌面投影。这是用简单反馈隔离
参考处理影响的**短窗口对照**，与先前质量优化控制策略不同。投影只覆盖 finger2
mesh 和桌面盒，净空 1 mm、最大抬升 5 cm、形变界为零；桌面盒标定取自本次仿真
场景，真实部署标定未完成。

| 指标 | 原目标 | 零位姿误差投影 | 0.01 m / 0.05 rad 包络投影 |
| --- | ---: | ---: | ---: |
| 完成新控制周期 | 7/7 | 7/7 | 7/7 |
| 原目标位姿双容差达标端点 | 2/7 | 2/7 | 2/7 |
| 所选目标位姿双容差达标端点 | 2/7 | 7/7 | 0/7 |
| 原目标位置误差均值 | 0.018082 m | 0.017627 m | 0.019013 m |
| step126 原目标位置误差 | 0.031719 m | 0.031181 m | 0.035987 m |
| 目标相对原参考的最大抬升 | 0 | 0.030631 m | 0.043554 m |
| finger2–桌面有效接触内步 | 156/175 | 156/175 | 81/175 |
| 任意 robot–environment 有效接触内步 | 156/175 | 156/175 | 133/175 |
| finger2–炉具燃烧区有效接触内步 | 146/175 | 146/175 | 123/175 |
| 任意机械臂力矩裁剪内步 | 0/175 | 0/175 | 0/175 |

接触计数表示某一 2 ms 内步是否至少存在该几何对的有效约束，不是接触点数；
不同几何对可同时接触，不能直接相加。以下按周期列出实际 finger2–桌面有效
接触内步，每周期共有 25 个内步：

| step | 原目标 | 零位姿误差投影 | 位姿包络投影 |
| --- | ---: | ---: | ---: |
| 120 | 6 | 6 | 6 |
| 121 | 25 | 25 | 25 |
| 122 | 25 | 25 | 25 |
| 123 | 25 | 25 | 25 |
| 124 | 25 | 25 | 0 |
| 125 | 25 | 25 | 0 |
| 126 | 25 | 25 | 0 |

位姿包络分支在后 3 个周期降低了桌面接触，但仍有大量炉具接触，且相对其
投影目标的 7 个端点均未达到原 `0.01 m / 0.05 rad` 双容差；不能将其提升为
成功控制方案。零误差候选虽然 7/7 端点达到自身投影目标的这组容差，实际桌面
接触数未减少，说明端点误差或静态目标无碰撞并不等于执行路径无接触。
三个分支的原生任务标签在这七个端点均为真、目标抓取标签均为假；这只是很短
窗口的瞬时标签，不是 600 周期稳定任务成功或物理安全结论。

每个内步在环境既有 forward 和控制之后、积分前采集 TCP、顶点、控制目标和
裁剪前后力矩；接触取同一内步积分后的求解缓存。三个分支最大实际 TCP 局部
顶点偏移均约 `0.003057 m`。位姿包络分支中，仅 7/175 个采样点满足**精确零**
形变条件，仅 1/175 个同时满足零形变、位置和旋转上界；不能把固定形状的静态
包络推广到其余内步。实际接触和跟踪还受其他几何、动力学、限幅与历史影响。

三次 producer 和三个不加载候选策略的 checker 都完成并实际 wait/reap；每次
127 个控制周期、3,175 个 2 ms 内步，合计 19,050 内步及 21 个新选择命令。
checker 逐项对比端点、关节内步、控制器目标、接触及力矩，匹配后通过；checker
重放的是 producer 已派发命令，并未独立重算其策略或停机决策。预检 32 并发
专项 29 项通过，最终 64 并发相关专项 36 项通过；实验及只读汇总审核均无新
WAM/GPU 调用或新初态。

原 owner 的 checker 汇总把不存在的策略偏移字段默认写成 `0`，这并非实际
投影目标偏移。原结果保留不改；[只读审核](../../../outputs/tabletop_projection_feedback_summary_audit_20261111/result.json)
从 producer 原目标及 checker 保存的所选目标重建偏移，分别得到最大 `0`、
`0.030631 m`、`0.043554 m`。新代码把 checker 的该策略统计标记为未计算，
并明确它不重新评估 producer 的停机理由。物理实验时 owner 脚本对应 Git 提交
`7efaa5cf`，历史声明路径为 `scripts/simulation/runtime/run_tabletop_projection_feedback_probe.py`，
SHA256 为 `15d674945c9e103220273fe5c6b2caccd70936d986c8fbc4cdd0335dce413763`。
为在 squash 后保留确切字节，已在[历史源码快照](../../../docs/history/experiments/tabletop_projection_feedback_owner_7efaa5cf.py.txt)
保存同 SHA 副本；同次预检涉及、后来更改的
`tests/research/native_runtime/test_tabletop_projection_feedback_probe.py` 也保存了
[历史副本](../../../docs/history/experiments/test_tabletop_projection_feedback_probe_7efaa5cf.py.txt)，
SHA256 为 `a08823124c60bd6953736ef2ba3d3f4d757b6d620bcfb630c0b259c54292fb01`。
其余实验来源身份与登记一致；后续汇总修正不覆盖历史来源或重新运行物理实验。

下一步应显式处理运动中的夹爪及桌面、炉具等相关几何，并在同一接口下对照
能跟踪修正参考的反馈方法。只有先解决实际反馈和参考偏差，才值得扩展开发
初态及使用新图像产生的世界模型参考；原 18/12/24 和每条 600 周期验收未完成。

| 仓库内证据 | SHA256 |
| --- | --- |
| [零工作 dry](../../../outputs/tabletop_projection_feedback_probe_dry_20261111/result.json) | `7d381ca2d04390f362d726549d335f8bd225ffdbd21de59c237bb6928368e9ae` |
| [32 并发准入专项](../../../outputs/qa/tabletop_projection_feedback_probe_20261111/result.json) | `f0bb34482d41d0975e63c22d99914f146ab1fdcb3a1d9f868cd6ea935c67ebeb` |
| [物理实验 owner](../../../outputs/tabletop_projection_feedback_probe_20261111/result.json) | `a3598795456067aeb66b6d23ec3a7cc26adcda07c690286cd12383f8c5069466` |
| [原目标 producer](../../../outputs/tabletop_projection_feedback_probe_20261111/original/producer/simulation/result.json) | `5ecbc3c50924d6c8056d0e6b17a80f288c14bae50da3189fd10c6fb06f2e1184` |
| [零误差 producer](../../../outputs/tabletop_projection_feedback_probe_20261111/nominal/producer/simulation/result.json) | `9e413ad91c188f97f6202751f954b43abd0eb7cd2f7085d652d4bf5585d5ba0c` |
| [位姿包络 producer](../../../outputs/tabletop_projection_feedback_probe_20261111/pose_bounded/producer/simulation/result.json) | `e2539dd739d27da9e9c109a2490a7e8f43f223a67362402645e09ef5f38c7e60` |
| [原目标 checker](../../../outputs/tabletop_projection_feedback_probe_20261111/original/checker/simulation/result.json) | `9a68af76610488f40ab03938982ea46b746346e38d826807b3d895d88670475e` |
| [零误差 checker](../../../outputs/tabletop_projection_feedback_probe_20261111/nominal/checker/simulation/result.json) | `620048f847631d1c4b538c905dff3cbb63ae7b86490ce2015307e16956f51698` |
| [位姿包络 checker](../../../outputs/tabletop_projection_feedback_probe_20261111/pose_bounded/checker/simulation/result.json) | `4b56dd074da242f2bc2849ba1e59a828d555f822144ca97c90c2ea770d41c7ad` |
| [原汇总的只读修正审核](../../../outputs/tabletop_projection_feedback_summary_audit_20261111/result.json) | `320a109b8bcfd171ef74f44626aac1abffda12b7bc26f7919909562d6745b858` |
| [最终 64 并发相关专项](../../../outputs/qa/tabletop_projection_feedback_final_20261111/result.json) | `bf9f8bc0c380f9127aa121387978c3b8a399c77aee065ef2159abee80a9eec7d` |
