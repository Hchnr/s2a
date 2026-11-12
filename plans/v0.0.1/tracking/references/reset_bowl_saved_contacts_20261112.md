# 碗任务已保存端点的指垫接触诊断

2026-11-12。两条同 reset、seed42 的碗任务开发轨迹各有 48 个已审计周期；固定 H40
分支较接近碗原点，但两条均没有记录目标 grasp。此处只检查已保存 reset 与周期端点
重建出的 MuJoCo 接触列表，帮助决定下一步的因果控制对照；不创建新轨迹，也不向
在线控制器提供目标状态。

固定输入为[已有几何恢复结果](../../../../reports/tracking/references/reset_bowl_saved_geometry_recovery_20261112.md)、
其原始物理轨迹与独立 checker、XML 和全部已记录资产。沿用该恢复的状态与指纹检查：
重建 1 个共同 reset 与两组各 48 个端点，TCP 捕获点最大误差 1e-9 m；仅允许已登记
的机器精度四元数归一化。不得改变已执行来源。逐端点记录对应目标接触对、几何名和
MuJoCo 的有符号 `dist`，左右指垫分别为 `gripper0_finger1_pad_collision` 与
`gripper0_finger2_pad_collision`。碗目标接触组是同一目标刚体下 group 0 的
`akita_black_bowl_1_g1` 至 `akita_black_bowl_1_g40`；`g0` 为 group 1 可视几何，
不计入。该选择对应 robosuite 的 `contact_geoms` group 规则及 Panda gripper 的
`important_geoms`，实现时必须验证 XML 和编译模型一致。

按 robosuite 的原生判据，两侧指垫各至少与一个目标 contact geom 出现在当前
`data.contact` 中才标记双侧接触；不增加力阈值。将此重建标记与保存的目标 grasp
标签逐端点比较；任何不一致均报告，不能静默改写标签。其余碗接触对单独记录，
以免将桌面或其他机器人几何误记为指垫。MuJoCo 可记录带 margin 的候选接触；列表、
`dist` 或双侧布尔值不等同接触力、动态稳定抓取，也不能证明中间 25 个物理内步的
情况，更不能从单个旧初态推出一般任务性能。

仅允许新的 1 个 CPU MuJoCo 模型和 data、最多 97 次 `mj_forward`、零物理步、零
渲染 context、零 WAM 初始化和请求；前两次几何诊断已使用 2 个模型/data、107 次
forward，本计划的累计上限为 3 个模型/data、204 次 forward。单个受监管子进程
实际 wait/reap，失败保留且不自动重试。先检查左右配对、顺序、目标组和原生接触
语义，运行真实 CPU MuJoCo 的小型测试、32 并发回归及零模型 dry，再执行诊断；
所有输入、代码和产物有 SHA 与实际资源计数。结果只属于离线诊断，不能改变原来的
任务验收门槛，也不能视为实时或真机证据。
