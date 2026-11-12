# 碗任务保存状态的指垫与其他几何接触

2026-11-12。按[预登记计划](../../../plans/v0.0.1/tracking/references/reset_bowl_saved_contacts_20261112.md)，
对同一旧开发初态、seed42 的[固定 H40 和逐 8 周期重观测](reset_bowl_fixed_h40_probe_20261112.md)
两条轨迹，各检查 48 个周期末保存状态；共同 reset 另检查一次。这里的接触是根据
保存状态重新 `mj_forward` 后的 MuJoCo 接触列表，不是新执行的抓取或接触力测量。

| 重建的周期端点 | 固定 H40 | 逐 8 周期重观测 |
| --- | ---: | ---: |
| 碗–左指垫接触列表非空 | 0/48 | 0/48 |
| 碗–右指垫接触列表非空 | 0/48 | 0/48 |
| 两侧指垫同时接触 | 0/48 | 0/48 |
| 与原记录目标 grasp 标签不一致 | 0/48 | 0/48 |
| 碗–指垫以外几何接触列表非空 | 48/48 | 48/48 |

最后一行包含碗–桌面接触，不能当作抓取。固定 H40 分支第 43–48 周期端点的列表
另有碗与 `gripper0_finger1_collision` 接触，第 46–48 周期端点另有碗与
`gripper0_hand_collision` 接触；两类最小 `dist` 分别约 −0.00236 m 和 −0.00154 m。
重观测分支的 48 个端点未出现碗与 `gripper0_` 几何的接触对。两个分支的共同前
16 个端点的重建接触列表完全一致。这些几何名和有符号距离不等于夹爪指垫、接触力
或稳定抓取；这些相关现象也不足以单独解释先前观察到的碗原点位移。

原生判据中的左右指垫分别是 `gripper0_finger1_pad_collision` 和
`gripper0_finger2_pad_collision`；碗 contact geoms 为 `akita_black_bowl_1_g1`–`g40`，
可视 `g0` 不计入。此处遵循原生 `data.contact` 是否包含两侧对的判据，不施加力或
距离阈值；MuJoCo contact list 可以包含 margin 内的接触候选。结果只覆盖保存的
周期末状态，不能排除周期中 25 个内步出现过接触。当前短窗中既未观察到指垫接触，
也未观察到抓取；不能只归因于夹爪开度，或推断改变闭合策略一定成功。

## 核验与资源

使用资产指纹核对过的同一 XML 新建 1 个 CPU 模型/data，对 reset 和 96 个端点共
调用 97 次 `mj_forward`，未推进物理、创建渲染 context 或请求世界模型。97 个状态
与前次重建的 TCP/物体原点及端点身份核对；8 个捕获时刻 TCP 最大绝对差仍为 0。
延续前次四元数归一化审计。此前失败、成功几何查询及本次累计为 3 个模型/data、
204 次 forward、零物理步/context/模型请求；本次子进程已实际 wait/reap。

诊断前 32 并发相关回归 56 项全部通过，0 skip；零模型 dry 通过。结果
`outputs/reset_bowl_saved_contacts_20261112/result.json` SHA-256
`c904ffc6a94d97986c1c1d59c8d9803cf0a74d3f8436ff0169be5557fa1d4912`。
[只读审计](../../../scripts/simulation/runtime/audit_reset_bowl_saved_contacts.py)逐端点
检查接触分组、双侧判据、原标签、共同前缀和上一几何结果，不重新构建模型或执行
forward；结果 `outputs/reset_bowl_saved_contacts_audit_20261112/result.json` SHA-256
`678b9e3d3a2e56cc89808912e359c6cb6b3f94d356e46ff2e2adf42c9df33b50`。

该分析仅属于一个旧初态的离线仿真诊断，不增加任务成功次数，不改变 600 周期留出
验收门槛，也不是实时或真机证据。后续需要设计只使用在线可用观测和历史的接近、
姿态及夹爪动作对照，并以独立物理重放和原任务成功指标判断。
