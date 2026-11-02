# WAM 发布标签与执行 TCP/时间合同核对（2026-10-20）

本次只读核对未发现已审计路径把旧发布 flange 当作 canonical TCP 的错误。只使用 CPU 读取已有文件，没有加载模型、启动仿真或改变主评测。它验证坐标与标签语义，不是新的任务成功证据。

原诊断对象是 `8800_masked/task1_init1_s43`、wine `demo_9`；来源审计没有找到它对应的精确训练 episode，因此不声称已定位该示范的训练标签。数值标签对照使用同任务、精确命令匹配的 `task1_init2/demo_4 → R1 episode683 → R2 episode1139`，共158帧、157条有效转移。1139也是旧v2直接state评测样本；该对照不能称为预训练未见数据。

原生 `grip_site` 位置配合 `right_hand` 轴，经 `source_from_tcp` 右乘到 `closed_pad_tcp`。旧发布参考点先经过另一套变换，因此其 +92.9mm 和原生点的 −3.6mm 不是同一起点的两个偏移。包含旋转和平移的完整合同满足：

```text
source_to_reference @ current_published_from_tcp == source_from_tcp
max_abs = 2.22e-16
```

| 核对 | 范围 | 最大位置误差 m | 最大 SO(3) 误差 rad |
| --- | --- | ---: | ---: |
| R2 state 与 R1 原生实测位姿经 source_from_tcp | 158帧 | 6.1014e-8 | 3.5941e-8 |
| R2 delta 与精确 canonical TCP 相邻帧变化 | 157转移 | 4.5672e-10 | 2.7279e-9 |
| 同 index 的 R2 state 与 HDF5 obs 经相同工具变换 | 158帧 | 0.013686 | 0.105933 |

第三行保留为来源/时间边界。HDF5的 `obs[t]` 关节和夹爪与保存的 `states[t+1]` 完全一致，属于原生动作后的观测；发布R1/R2观测是另一条上游流。比较相邻 ±1 索引仍不能得到精确位姿一致，没有拟合偏移、重采样或以此修改执行锚点。不得把第一行的精度宣称为对HDF5原生流的重建精度。

发布行 `t` 的 state 是当前 TCP，pose action 是 `inverse(TCP[t]) @ TCP[t+1]`；opening action 是下一观测的归一化 pad aperture。末行的pose/opening/mask均无有效转移。1139的 action slot9 与下一观测开度最大差为2.8293e-8，与当前开度最大差0.121361；它是端点状态标签，不是原生开关或力命令。冻结S2A仍需把该参考变成实际夹爪命令，其执行局限不能由合同一致性排除。

原请求轨迹 `task1_init1_s43` 的75个chunk锚点均严格等于当前 `measured_source @ source_from_tcp`，600个执行目标转回source后均与日志严格相等（max_abs=0）。结合上述同任务标签核对，未发现这条路径的工具点重复转换或错用旧flange；没有证明demo9的精确训练来源，也没有唯一解释它的任务失败。

当前训练reader直接消费发布action、mask和timestamp，不消费state条件，也不在reader内应用上述工具矩阵。当前catalog、receipt和checkpoint配置不能证明历史训练实际消费的字节及窗口；消费记录未提供，不将静态合同核对提升为历史训练全链认证。

原始数值、索引比较、41项输入SHA及代码依据见 [audit.json](../outputs/wam_tcp_time_alignment_20261020/audit.json)，产物SHA256为 `da7f2f02215d8ae496608bfae562c7f8cee6e9631ea140b10ac81f8e1764b51e`。执行锚点见 [checkpoint_simulation.py:155](../src/s2a/checkpoint_simulation.py#L155)，原生观测点见 [libero_simulation.py:58](../src/s2a/libero_simulation.py#L58)，完整来源边界见原始审计的 `limitations`。

后续只读第一方核对已认证该同源流对应[NVIDIA固定发布版本](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3/tree/e5907374380b8f96511957e6ba5582be52a1e179)：两份本地元数据文本原字节、四个goal数据/视频shard SHA与远端内容身份相同。原始NVIDIA生成器、HDF版本和图像/pose物理同步仍未闭合；这项新证据不改上表的TCP/时间数值，也不证明历史训练消费。第一方响应及边界见[图像/时刻报告](wam_video_pose_timing_20261020.md)与[audit_reconciled.json](../outputs/wam_upstream_generation_contract_20261020/audit_reconciled.json)。
