# Franka 跨数据集验证扩展

用户要求：说明当前有效性的证据、转换方式与机器人型号，并在其他数据集的相同型号上试验。当前没有执行过硬件；基线来源是 DROID 的 Franka Emika Panda + Robotiq 2F85 日志。新增工作只使用已授权本地只读数据，不把日志复核或模型推进写成真机执行。

## 固定方案

- 解析部分不训练：`T_BC_next = T_BC_now @ Delta_T_C`，再通过各数据集自己的 TCP 外参还原原始 source pose；保持 column-rot6d、SE(3) 复合与米/弧度约定。
- 数据集：RoboMIND-v1 的 Panda/Robotiq 分支 20 条、InternData-M1 的 Panda 仿真分支 20 条；另取 RoboMIND-v1 Panda 原生工具 10 条，区分同机械臂与不同末端工具。
- InternData-M1 20 条与 InternData-A1 Robotiq 10 条均作为仿真日志补充，明确不计入真实机器人数据结果。RoboMIND2.0 存在 Panda/FR3 混合及名义 FK 定义，不纳入“同型号”比较。
- seed=42，各 cohort 仅依 metadata 在 training_eligible=true、同一工具 contract、length>=40 的 episode 中均匀无放回抽样。名单先写入 `configs/cross_dataset_franka.json` 并提交，再打开数值结果；不因误差替换样本。
- 每条完整轨迹独立评估。位置/姿态容差沿用 `1e-5 m / 1e-4 rad`；无效 transition、尾行和不连续 frame 不积分，连续片段各自锚定。
- 报告单步、累计、原生观测 pose 交叉检查和保持当前 pose 基线；有独立保存的 Cartesian action 字段时单列差异，但不将其当已确认的机器人接口。
- M1 原始 effector 方向到 link8 有固定 z+45°，依据发布器 `round2_op/publish/eef_state_sources.py:119`；RoboMIND-v1 使用 contract 的 `link8_from_canonical_tcp`（同文件89行）；A1 native quaternion 是 wxyz。复制数学约定，不 import 外部发布器。
- 每条数据使用 provenance 的 native_binding episode + row_range/frame 对齐；不用相近 timestamp join。保存原始 masks、媒体时间、contract、receipt、样本及代码/文件 SHA256。
- 这些分支未提供 DROID 已核验的机器人物理时钟列，历史控制器与限幅未确认；不借用媒体帧率生成 SI 速度，不转移 DROID 缩放/B/C/PID，不虚构跨数据集 closed-loop 分数。

## 可解释的终点

解析几何通过只说明表示与坐标合同正确；生成的候选 source pose 是否经原控制器达到目标仍未验证。相同 Panda 型号不代表同序列号、夹爪、TCP 或控制配置。真实跨机器人控制泛化需要独立执行系统；本轮报告其缺失条件，不把缺少的数据补成理想动力学。

公开型号证据：[RoboMIND 官方项目](https://x-humanoid-robomind.github.io/)；[InternData-A1 官方项目](https://internrobotics.github.io/interndata-a1.github.io/)。[M1 官方数据卡](https://huggingface.co/datasets/InternRobotics/InternData-M1/blob/main/README.md) 明确该分支为仿真；M1/A1 不计入真实机器人数据。

补充诊断：RoboMIND 保存 master/puppet joint；使用已有名义 Panda URDF 只计算 FK，并核对 puppet FK 与原始 EEF。master FK target 与候选 pose 的差异仅作命令代理诊断，不做 IK、不据此假设真实控制器相同。
