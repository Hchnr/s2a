# S2A 001 — 从期望状态反解控制命令

首次编写：2026-10-16（Asia/Shanghai）。源 Cosmos 代码快照：`7e124b9725c8d01860d30b5824e677bd4b7b10d6`。

本计划从 Cosmos 仓库的 `plans/v0.4.0_s2a/plan.md` 迁移至独立仓库 `/share/project/eai_pwm/home/hcr/repos/s2a`。**后续实现、测试、配置、报告和实验产物均在 S2A 仓库下维护。** 下文引用的 Cosmos 源码仅作为预训练数据语义的外部证据。

初始状态（实施前）：计划已编写，已完成只读的数据定位与单 episode 字段核查。

2026-10-16 实施更新：P0–P3 的代码、测试和离线实验已完成；几何检查通过，但控制候选未达到全部稳健验收条件，P4/P5 未执行。原始研究设计保留如下。过程与决策见 [实施记录](record_20261016.md)，证据与未执行项见 [结论](conclusion_20261016.md)。

本次修订：明确 S2A 是控制器及机器人响应的条件逆问题；解析转换、线性辨识、P/PI/PD/PID、增益调度和轻量网络均列入待比较方案，按数据与实验依赖推进，不预先指定哪类方法胜出。当前完成的是计划迁移和项目基础结构；各方案实现与性能验证尚待开展。

**1. 要验证的判断**

用户意图对应以下两条流程：

```text
普通 VLA：观测/任务 → VLA → action → 固定控制器 → 机器人/环境 → 实际 state
世界模型：观测/任务 → 世界模型 → 期望 state / delta_state
                                  → S2A 反解 action
                                  → 同一固定控制器 → 机器人/环境 → 期望接近的 state
```

用 `F_C` 表示固定控制配置 C 下“控制器 + 机器人/环境”的前向响应：

```text
s_(t+1) = F_C(s_t, u_t, history_t, dt)
u_t = G_C(s_t, desired_state_or_trajectory, history_t, dt)
目标：F_C(s_t, u_t, history_t, dt) ≈ desired_state_(t+1)
```

因此 S2A 的主要指标是命令经过同一前向系统之后的状态误差，而非只要求预测命令等于历史日志中的命令。历史 action 用于监督和诊断；存在多个可行 action 时，以达到目标状态、命令平滑和接口约束作为选择依据。控制器内部未观测状态由可用历史近似；不把该近似写成已知完整动力学。

在本项目中，模型预测的 state/delta_state 是被跟踪的目标；机器人实时测得的 state 是反解与反馈的条件。当前 Action128 的 pose delta 要先解码并积分成目标轨迹。PID 是 S2A 的一种闭环实现组件，也可以与解析逆或学习逆组合；项目范围不限于 PID。

这个方向具备可执行的起点：让世界模型负责预测希望发生的状态变化，由一个针对固定机器人及控制接口的小模块，把预测转成可发送的控制命令。第一版先冻结世界模型，使用训练数据中的真实状态变化作为输入，隔离验证转换本身。

更准确的形式是：

```text
(u_t, z_next) = adapter(s_t, desired_delta_state[t:t+H], z_t, controller_config)
```

其中 `u_t` 是控制器 API 接收的命令，`s_t` 是当前可测状态，`z_t` 保存因果历史、参考轨迹游标、积分量、微分滤波状态及上一条命令。第一版的输出层级是 Cartesian 目标/控制增量加夹爪目标，暂不涉及电机电流或关节力矩。

需要区分三个层次：

- **表示转换**：局部位姿增量 → 基座坐标系目标位姿 → 控制器输入格式。约定已知时，可以解析完成。
- **前馈补偿**：根据期望运动及固定接口的响应规律，提前构造候选命令；线性辨识可提供局部近似。
- **闭环反馈**：持续比较目标轨迹与真实反馈，用 P/I/D 修正偏差。实际响应依赖速度、延迟、控制器内部状态和接触，不能仅由坐标变换保证。

训练中的 observed delta 是已经发生的运动；推理时我们才把预测结果当作期望运动。单独的 `delta_state` 通常不能唯一确定原控制命令，尤其存在惯性、阻抗、饱和和冗余关节时。因此本实验验证的是“固定条件下是否能构造有效、足够简单的 adapter”，而非假设存在全局唯一的逆函数。

**2. 原 Cosmos 训练链路与数据已经确认的事实**

本节的代码定位属于外部参考仓库 `/share/project/eai_pwm/home/hcr/repos/test/cosmos-framework-pwm`，不表示 S2A 包依赖其训练代码。数据预检结果保留首次核查日期及源代码快照，实施前重新固定数据文件指纹。

| 事实 | 证据及影响 |
| --- | --- |
| 当前 160 GPU 配方包含 `Cosmos3-DROID-eef` | [examples/toml/sft_config/action128_wam_nano_160gpu.toml:139](/share/project/eai_pwm/home/hcr/repos/test/cosmos-framework-pwm/examples/toml/sft_config/action128_wam_nano_160gpu.toml:139)，catalog 为 20261015 发布版本。这里的“当前训练数据”以这份配方为依据；没有检查线上进程实际消费记录。 |
| Action128 reader 读取发布的 action，执行 source-domain affine normalization 和 clipping | [cosmos_framework/data/generator/action/unified_action128.py:601](/share/project/eai_pwm/home/hcr/repos/test/cosmos-framework-pwm/cosmos_framework/data/generator/action/unified_action128.py:601)、`:614`。实验从物理/source 单位的 Parquet 开始，避免重复反归一化及截断损失。 |
| action_dim_mask 与 action_control_mask 是不同概念 | [cosmos_framework/data/generator/action/unified_action128.py:622](/share/project/eai_pwm/home/hcr/repos/test/cosmos-framework-pwm/cosmos_framework/data/generator/action/unified_action128.py:622)；[docs/action128_training.md:8](/share/project/eai_pwm/home/hcr/repos/test/cosmos-framework-pwm/docs/action128_training.md:8)。有效训练目标不一定是控制命令。 |
| 当前 Action128 WAM reader 不接收 state 条件 | [cosmos_framework/data/generator/action/unified_action128.py:474](/share/project/eai_pwm/home/hcr/repos/test/cosmos-framework-pwm/cosmos_framework/data/generator/action/unified_action128.py:474)。adapter 可以独立读取机器人状态，无须先修改世界模型训练接口。 |
| 仓库已有 framewise 相对位姿的积分工具 | [cosmos_framework/data/generator/action/utils/pose_utils.py:420](/share/project/eai_pwm/home/hcr/repos/test/cosmos-framework-pwm/cosmos_framework/data/generator/action/utils/pose_utils.py:420)、`:443`、`:472`。可用来核对数学约定。 |
| 旧 RoboLab 服务已有相对位姿 → 绝对位姿的后处理 | [cosmos_framework/scripts/action_policy_server_robolab.py:606](/share/project/eai_pwm/home/hcr/repos/test/cosmos-framework-pwm/cosmos_framework/scripts/action_policy_server_robolab.py:606)、`:614`。它证明已有几何后处理入口，但不能证明当前 Action128 TCP 契约或实际控制响应已被解决。 |

首个数据对象固定为 **DROID / Franka Panda + Robotiq 2F85**。只使用一个 episode，避免一开始混合不同机构、设备及控制器设置。episode 来自 AUTOLab；具体机器人序列号与历史控制器版本仍需追溯，不能把整个 DROID 数据集视为同一台设备。

以下路径作为本计划的数据依据：

```text
CATALOG = /share/project/liyue/data/parquet-data/action128-full-20261015/Cosmos3-DROID-eef/Cosmos3-DROID-eef.parquet
R2      = /share/mixed/projset/eai_pwm/datasets/action_processed_r2/Cosmos3-DROID/eef
R1      = /share/mixed/projset/eai_pwm/datasets/action_processed_r1/Cosmos3-DROID/success
```

catalog 中 `data.root=R2`，`adapter=unified_action128`，`use_state=false`，训练窗口和 stride 均为 2 秒。选中 episode 的完整 action domain 为：

```text
franka_panda__parallel_gripper/eef20/observed_local_delta/droid_robotiq2f85_closed_pad_observed_eef10_to_action128_v1
```

本次读取到的契约：

- `action[0:3]` 是当前 canonical TCP 局部坐标系的平移增量；`action[3:9]` 是相对旋转矩阵前两列的 column-rot6d。
- `observation.state[0:9]` 是当前 TCP 在机械臂基座坐标系中的绝对位姿；slot 9 是当前夹爪 opening。
- **当前 `action[9]` 已经是原生夹爪绝对命令转换后的 opening，不是机械臂 delta，也不是下一帧夹爪观测。** 本次 episode 的 control mask 只在 slot 9 为真。
- 其余槽位没有有效目标。episode 末行仍有夹爪命令，因此 `action_valid=true` 不代表末行具有有效位姿转移；必须同时检查 pose 的 dim mask 和 physical_time_valid。
- 物理时间使用 `source_physical_timestamp / transition_dt`；`timestamp` 保留视频时间轴。不可把视频帧率直接当实测控制周期。

这些判断来自当前契约、当前 receipt 和实际数据，不能用历史 repair 记录中的旧夹爪描述覆盖。准确文件定位如下：

| 依据 | 文件位置 |
| --- | --- |
| state 布局、参考系与边界规则 | [eef_state_contract.json:3](/share/mixed/projset/eai_pwm/datasets/action_processed_r2/Cosmos3-DROID/eef/meta/eef_state_contract.json:3)、[eef_state_contract.json:33](/share/mixed/projset/eai_pwm/datasets/action_processed_r2/Cosmos3-DROID/eef/meta/eef_state_contract.json:33) |
| episode 0 当前基座参考系、夹爪来源 | [_receipts/data/chunk-000/file-000.json:14](/share/mixed/projset/eai_pwm/datasets/action_processed_r2/Cosmos3-DROID/eef/_receipts/data/chunk-000/file-000.json:14)、[同文件:24](/share/mixed/projset/eai_pwm/datasets/action_processed_r2/Cosmos3-DROID/eef/_receipts/data/chunk-000/file-000.json:24) |
| canonical TCP 与原始 flange 的转换 | [droid_robotiq2f85.json:21](/share/mixed/projset/eai_pwm/datasets/action_processed_r2/Cosmos3-DROID/eef/meta/eef_tools/droid_robotiq2f85.json:21)、[同文件:29](/share/mixed/projset/eai_pwm/datasets/action_processed_r2/Cosmos3-DROID/eef/meta/eef_tools/droid_robotiq2f85.json:29) |
| 物理时间与媒体时间的区别 | [eef_time.json:3](/share/mixed/projset/eai_pwm/datasets/action_processed_r2/Cosmos3-DROID/eef/meta/eef_time.json:3) |
| R2 到 R1 的 episode/row 来源 | [processing/chunk-000/file-000.jsonl:1](/share/mixed/projset/eai_pwm/datasets/action_processed_r2/Cosmos3-DROID/eef/processing/chunk-000/file-000.jsonl:1) |

**3. 第一段数据与最小数据包**

选择 R2 `episode_index=0`，对应 R1 success `episode_index=0`：

```text
source episode_id: AUTOLab/success/2023-07-07/Fri_Jul__7_09:42:23_2023
task: Pour contents from yellow cup into pink bowl
R2 episode metadata: R2/meta/episodes/chunk-000/file-000.parquet
R2 numeric data:     R2/data/chunk-000/file-000.parquet
R1 numeric data:     R1/data/chunk-000/file-000.parquet
selected transitions: frame_index ∈ [30, 330)
required observations: frame_index ∈ [30, 330]，另保留最多 3 帧左侧历史
```

只读核查结果如下；这些是数据预检结果，不是 adapter 实验成绩：

| 项目 | 本次读取结果 |
| --- | --- |
| 全 episode | 471 行；470 个有效位姿转移；471 个有效夹爪命令 |
| 选段 | 300 个转移；位姿 mask 与 physical_time_valid 均有效 |
| 选段媒体时间 | `[2.0, 22.0]` 秒，endpoint observation 包含在内 |
| 选段物理时间 | `[2.106992219, 23.339967628]` 秒，跨度 `21.232975409` 秒 |
| 全 episode 正 transition_dt 的 min / median / max | `0.067012293 / 0.070968930 / 0.074027217` 秒 |
| 行映射 | R2 source_frame_index、raw_frame_index 与 R1 frame_index 逐行一致 |
| 当前夹爪命令关系 | `R2.action[:,9] = 1 - R1.action.gripper_position`，全 episode 最大绝对误差约 `2.98e-8` |
| 当前夹爪状态关系 | `R2.state[:,9] = 1 - R1.observation.state.gripper_position`，最大绝对误差约 `2.98e-8` |

R1 中已确认保留独立的原始命令列：`action.cartesian_position`、`action.cartesian_velocity`、`action.joint_position`、`action.joint_velocity`、`action.gripper_position`。另有 `observation.state.cartesian_position`、`joint_positions`、`joint_velocities` 和夹爪状态。因此可以比较 observed delta 与真实记录的命令，不必把重建出的下一状态冒充 action 标签。旧 reader 的相关字段定义可交叉参考 [cosmos_framework/data/generator/action/datasets/droid_lerobot_dataset_config.py:103](/share/project/eai_pwm/home/hcr/repos/test/cosmos-framework-pwm/cosmos_framework/data/generator/action/datasets/droid_lerobot_dataset_config.py:103)。

实施时输出 `sample_manifest.json` 和 `sample.npz`，保存：

- catalog/root、完整 domain、代码 commit、源文件 SHA256、episode ID、frame IDs、列名、两套时间轴、所有相关 masks。
- raw Action128、当前 TCP 位姿、原始 flange 位姿、关节位置/速度、原始控制命令、夹爪状态及命令。
- TCP 外参、旋转编码、单位、坐标系、控制接口配置和其证据来源；未知值明确标记。

只读取所需列和 episode，不解码全量视频、不启动训练 dataloader、不修改 R1/R2 发布数据。按 frame/provenance 做 join，不能仅凭相近 timestamp 配对。

**4. 解析方案 A：零训练的几何与接口转换**

定义 `T_BC(t)` 为基座 B 到 canonical TCP C 的绝对位姿，`Delta_T_C(t)` 为发布的局部相对变换；`H_FC` 为原始 flange F 到 canonical TCP C 的固定变换。

```text
T_BC_desired(t+1) = T_BC(t) @ Delta_T_C(t)
T_BF(t)          = T_BC(t) @ inverse(H_FC)
T_BF_desired     = T_BC_desired(t+1) @ inverse(H_FC)
```

`H_FC` 从当前 [meta/eef_tools/droid_robotiq2f85.json:29](/share/mixed/projset/eai_pwm/datasets/action_processed_r2/Cosmos3-DROID/eef/meta/eef_tools/droid_robotiq2f85.json:29) 读取，包含绕 z 的轴转换及 0.1438 m 的 nominal TCP 偏移。它是发布数据采用的标准装配定义，不是历史每台设备的精确标定。不能只旋转平移向量而遗漏 TCP 力臂带来的位移。

按下列顺序验证：

1. 解码 column-rot6d 为 SO(3)，先核查训练真值的正交性。对未来模型输出，执行明确的正交化，并记录退化旋转输入。
2. 用单步 `T_BC(t) @ Delta_T_C(t)` 重建 `T_BC(t+1)`；再从首帧连续积分整段，分别报告单步及累计误差。
3. 用 `H_FC` 转回原始 flange，与 R1 当前观测位姿逐帧交叉核对。这里比较的是观测，不是原控制目标。
4. 形成 `cartesian_position` 格式的候选命令；DROID 的旋转格式以接口实现为准，不能直接传 rot6d 或 quaternion。
5. 如采用 DROID `cartesian_velocity` 接口，继续使用它的 `pose_diff` 与缩放规则编码，不能按字段名字把它当作 SI 单位速度。

公开 DROID 实现中，Cartesian “velocity” 会被转换成增量，再经 IK 形成 joint target；夹爪可单独选择 position 模式。实现阶段先保存并固定源码版本，再核对其是否匹配这段历史数据。[DROID robot.py:42](https://github.com/droid-dataset/droid/blob/main/droid/franka/robot.py#L42)、[robot.py:171](https://github.com/droid-dataset/droid/blob/main/droid/franka/robot.py#L171)

作为**待核对的接口基线**，当前公开实现给出的编码可写为：

```text
e_p = p_BF_desired - p_BF(t)
e_R = EulerXYZ(R_BF_desired @ transpose(R_BF(t)))
u_arm_A = concat(e_p / d_max_translation, e_R / d_max_rotation)
u_gripper = 1 - predicted_gripper_open
```

公开实现的 `d_max_translation=0.075`、`d_max_rotation=0.15`，平移与旋转分别做向量范数限制；旋转差采用旋转复合后转 EulerXYZ，而非 Euler 角逐元素相减。以上是接口编码规则，不能当作“实际末端一定移动这么多”的物理规律。[robot_ik_solver.py:8](https://github.com/droid-dataset/droid/blob/main/droid/robot_ik/robot_ik_solver.py#L8)、[robot_ik_solver.py:63](https://github.com/droid-dataset/droid/blob/main/droid/robot_ik/robot_ik_solver.py#L63)、[transformations.py:34](https://github.com/droid-dataset/droid/blob/main/droid/misc/transformations.py#L34)

若发送 6D Cartesian velocity 加绝对夹爪命令，必须显式设置 `gripper_action_space="position"`；公开接口的默认值会随 arm velocity 模式变为 gripper velocity。[robot.py:176](https://github.com/droid-dataset/droid/blob/main/droid/franka/robot.py#L176)

在使用这些常数前，以原始 `action.cartesian_position / cartesian_velocity` 和对应 state 做代数一致性检查。若不一致，优先检查历史版本、命令构造时刻的 `action_dict.robot_state` 与观测时刻是否不同、坐标系及限幅，而不是通过自由拟合“吸收”定义错误。未找到历史配置时，只能标记为 reference-interface baseline。

对整段参考轨迹，从锚点逐步积分目标；真正执行时使用最新反馈跟踪该参考轨迹，重新规划时再重设锚点。不能在每次反馈上盲目累加整段剩余 delta，造成目标漂移。第一轮仅计算候选命令及离线误差。

**5. 方案 B：解析/校准前馈 + 有状态的闭环 PID**

本节改为**解析/校准前馈 + 闭环 PID 系列**。固定增益、未限幅的标准 PID 是线性动态控制器；相较单帧 gain/bias，它增加了误差历史和反馈机制。是否足够取决于固定机器人的闭环响应，并非增加 I/D 就必然改善所有任务。先比较 P、PI、PD、PID，再决定保留哪些项。

**5.1 先生成带时间的参考轨迹，明确控制层级**

```text
世界模型 delta_state chunk
  → SE(3) 积分、时间对齐与插值 → 参考轨迹、参考速度 → 前馈 + PID
                                                        ↓
实测 state ← 机器人及已有底层控制器 ← 固定接口编码、限幅 ← 候选命令
```

世界模型以 chunk 更新参考轨迹，外层反馈按可获得的新鲜 state 更新，机器人原有底层伺服保持其职责。外层反馈频率不绑定世界模型推理频率，也不能从 15 Hz 历史数据推断真实硬件支持的最高反馈频率。

公开 DROID 后端会启动 Cartesian impedance 并更新 desired joint positions，因此本方案面对的是“原控制器 + 机器人”的整体对象。需要先确认历史/部署模式、延迟和限幅，再选择外层增益与更新周期；避免把已有内环当成无动态的理想速度源。[DROID robot.py:69](https://github.com/droid-dataset/droid/blob/main/droid/franka/robot.py#L69)

参考轨迹在固定基座系中连续维护。一次 chunk 内按目标轨迹计算参考点，不能每个反馈周期都重新以当前实测状态为起点、抹掉积累的跟踪误差。新 chunk 根据真实锚点和时间戳接入，做参考轨迹衔接；插值优先保证速度连续，以减少重规划时的命令突变。

这种任务空间“期望运动前馈 + 位姿误差反馈”的结构可参考《Modern Robotics》11.3；其速度控制示例采用 PI，说明本实验也应保留 PI 候选，而非预设完整 PID 最优。[Modern Robotics：任务空间前馈与 PI](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-3-motion-control-with-velocity-inputs-part-3-of-3/)

**5.2 外层误差与 P/I/D：作用于跟踪偏差**

第一版在转换后的 flange 上控制，参考和实测均使用同一固定基座坐标系。令 `k` 为外层反馈步，参考在当前时刻 `t_k` 取值：

```text
e_p[k] = p_ref(t_k) - p_measured(t_k)
e_R[k] = Log(R_ref(t_k) @ transpose(R_measured(t_k)))^vee
e[k]   = concat(e_p[k], e_R[k])

dt_feedback[k] = t_k - t_(k-1)
I_trial[k] = I[k-1] + e[k] * dt_feedback[k]
d_raw[k]  = (e[k] - e[k-1]) / dt_feedback[k]
alpha[k]  = exp(-dt_feedback[k] / tau_D)
d_f[k]    = alpha[k] * d_f[k-1] + (1-alpha[k]) * d_raw[k]

nu_cmd[k] = nu_ff[k] + Kp @ e[k] + Ki @ I[k] + Kd @ d_f[k]
```

这是小误差范围的任务空间 PID 候选式；`nu_cmd` 的平移部分为 m/s、旋转部分为 rad/s。平移与旋转分别设增益和尺度，首版采用对角矩阵（最多 18 个 P/I/D 增益）。要求 `dt_feedback > 0`、`tau_D > 0`；首次更新先初始化滤波器，异常时间间隔不直接进入差分。积分量 `I[k]` 是经过 anti-windup 处理的 `I_trial[k]`。旋转使用 SO(3) 误差及连续分支，不对 rot6d 或 Euler 角逐元素相减；接近 π 的分支变化需单独处理。

- P 纠正当前位姿偏差。
- I 积累实际跟踪误差，用来补偿持续偏差；只有确实存在稳态偏差时再启用。
- D 提供误差变化反馈，需滤波，并评估噪声与延迟导致的副作用。

**不能令 `e[k] = predicted_delta_state[k]` 并把它累加为 PID 积分项。** delta 定义目标运动；PID 的积分是机器人没有跟上目标的误差积累。以上 PID 离散时间定义及积分饱和处理参考 [ROS control_toolbox](https://control.ros.org/jazzy/doc/control_toolbox/doc/index.html#pid-controller)。

重规划可能引入参考跳变，D 项不能直接放大这个跳变。除上述“平滑参考 + 滤波误差微分”外，保留 derivative-on-measurement / 二自由度 PID 候选；参数选择固定在 validation 阶段。若使用参考速度与观测速度之差替代姿态误差导数，须明确其小角度近似和一致坐标系。[MathWorks：设定值权重与 derivative kick](https://www.mathworks.com/help/simulink/slref/two-degree-of-freedom-pid-control-for-setpoint-tracking.html)

**5.3 物理控制量到 DROID 命令的桥接**

`nu_cmd` 不直接等于 DROID 名为 `cartesian_velocity` 的归一化命令。零校准版本先按计划发送的命令时域 `dt_command` 构造短期目标：

```text
p_command = p_measured + nu_cmd[:3] * dt_command
R_command = Exp(hat(nu_cmd[3:] * dt_command)) @ R_measured
u_arm = encode_droid_pose_target(p_command, R_command, measured_state)
```

编码复用第 4 节核对过的 `pose_diff`、尺度、旋转约定和限幅。`dt_command` 是当前时刻已知的命令时域；`dt_feedback` 是已发生的观测间隔，两者分别记录。在线不能读取数据里下一观测才知道的 `transition_dt[t]` 作为当前输入。离线 oracle 可用完整物理时间轴定义目标；接入模型输出时，必须显式定义模型时间轴到期望执行时间轴的映射。

第一版前馈 `nu_ff` 来自参考轨迹的速度；后续可经局部辨识校准。桥接层保证单位与接口语义正确，实际执行速度是否跟随仍由闭环实验验证。对新的状态更新，重新计算反馈和命令；没有新反馈时不反复累加积分。

**5.4 控制器状态、限幅与整定顺序**

1. 先验证纯前馈与前馈 + P；随后分别比较 PI、PD，再检验完整 PID。外层增益在确认响应方向后施加非负、有界约束；不凭单段数据直接给出通用增益数值。
2. 保留平移/旋转向量限幅、命令变化率限制，记录编码前后实际送出的命令。积分先采用限幅及条件积分：输出饱和且误差继续推动饱和时暂停相关积分。若实现 back-calculation，使用最终限幅后命令在同一物理坐标中的差值；不可只看 PID 内部限幅而忽略下游饱和。[ROS anti-windup](https://control.ros.org/jazzy/doc/control_toolbox/doc/index.html#anti-windup-strategies)
3. 保存每个运行实例的积分、微分滤波、参考游标和上次反馈时间，跨 chunk 延续。新 episode、控制模式改变、时间倒退或重新初始化时按明确策略重置；正常重规划采用平滑切换，避免每个 chunk 清空积分。
4. 陈旧观测、掉帧和任务切换时定义积分冻结/恢复规则。接触阶段单独报告；没有接触力观测时，不能让未消除的位置误差无限推高积分命令。
5. 可训练部分优先是这些少量增益、滤波常数和前馈尺度。固定增益不够时，再考虑有界的状态相关增益调度；不把增益约束误认为稳定性证明。

**5.5 原线性辨识方案保留为辅助前馈模型**

方案 A 的几何检查通过后，可先检验每轴 gain/bias，再逐步加入固定延迟、少量历史项和 6×6 耦合矩阵，为闭环整定提供响应近似。它也作为“仅前馈”的对照组，不作为完整反馈控制的替代。

以 flange 在固定基座系中的观测运动构造响应量，例如平移速度及旋转 `Log(R(t+1) @ R(t)^T) / transition_dt`。这里的物理速度与 DROID 命令编码分开命名，不能混为同一量。对固定控制模式建立局部模型：

```text
y_(t+lag) = B @ u_t + C @ h_t + b + residual
```

`h_t` 逐步加入当前关节位置/速度、过去 1–3 步命令或运动量；只在有验证收益时增加输入。`lag` 初始搜索 0–5 个源采样步，并同时报告物理毫秒数。对不均匀时间间隔保留真实 dt，不重写视频时间轴。

固定 lag 和特征后，用 ridge regression 闭式求解参数。给定期望响应 `y*`，先计算带解析基线先验的正则逆解：

```text
u_calibrated = inverse(B.T @ W @ B + lambda * I)
      @ (B.T @ W @ (y* - C @ h_t - b) + lambda * u_A)
```

`W` 平衡米与弧度的尺度，尺度只从 fit 段确定。实现中使用线性求解器而非显式求 inverse。记录矩阵秩、条件数、lambda、残差和限幅比例；低激励方向不能宣称已识别。需要加入命令/变化率约束时，使用小规模约束最小二乘，不能把 clipping 后的结果称为无约束逆解的等价解。

延迟补偿所需的未来目标来自世界模型已经生成的参考轨迹；运行时不能读取未来真实观测。离线 oracle 实验允许用真实未来轨迹构造期望输入，但必须明确标识，后续用模型预测替换。前向辨识和延迟估计中的未来观测仅作为训练标签。

夹爪在当前数据中已有命令标签，第一版保留解析反向编码和原有夹爪控制器，不与 arm 的 6D 位姿 PID 共用误差或积分。如果以后换成仅有夹爪观测的数据，再单独考虑迟滞或开合状态机。

若单段的激励不足，先完成这个可观察结论，再扩展同一设备/控制配置的少量 episode；不能为了得到漂亮拟合结果混入其他机器人的数据。

**5.6 离线数据能支持什么：不能用同一条轨迹虚构闭环结果**

若参考轨迹由这段观测 delta 积分得到，同时又把同一段观测逐帧作为反馈，则同时间的跟踪误差近乎为零，无法识别有用的 P/I/D 增益。用下一帧目标减当前观测得到的是期望运动，也不能借此证明具备纠偏能力。更改命令后仍强行回放原观测，只能检查代码及命令格式，不是反事实闭环执行。

先用原始 state/action 对齐数据辨识局部前向模型，并在留出段验证其一步和多步响应；随后在该模型上运行候选控制器，使每一步新状态由候选命令推进。对初始偏差、响应增益、延迟、恒定扰动和观测噪声设置明确的扰动实验。它们是模型内实验，必须与匹配控制接口的独立仿真/后续真机验证分开标记；同一辨识模型上的自洽不能证明真实闭环稳定。

gain 搜索以闭环位置/姿态跟踪误差、稳态偏差、超调、命令变化率和饱和时间为目标，原始 command RMSE 作为辅助诊断。局部模型不可靠或激励不足时，先补同设备数据或控制接口信息，不强行整定 PID。

**6. 轻量可训练方案 C：增益调度与小网络对照**

先完成少量 PID 参数整定及 P/PI/PD/PID 对照，再在同设备的多条轨迹上比较有界增益调度、小型直接逆模型和残差逆模型。它们均是计划内实验，不以 PID 失败为启动条件；若单段激励或样本量不足，则补足同设备数据后再评估。残差版本在同一物理运动量空间组合：

```text
nu_cmd = nu_ff + PID(tracking_error, controller_state)
         + small_MLP(current_state, desired_motion, causal_history)
u = fixed_interface_encoder(nu_cmd)
```

候选为两层 64 hidden 的 residual MLP，参数预算低于 50K，输出仅 6D arm residual，初始残差为零。训练时通过固定接口编码与 R1 原始命令比较，校验物理残差与命令的单位；原始命令监督仅作初始化/辅助目标，闭环改进须通过 rollout 验证。所有分量相加后统一限幅并纳入 anti-windup。世界模型冻结；训练、验证、测试按 episode 划分，不能用 300 帧训练并声称模型具有泛化能力。

直接逆模型以当前 state、期望运动和因果历史预测固定接口的 arm command，作为“无需手写逆模型”的轻量对照；使用与残差版本相近的参数预算及相同数据划分。夹爪仍复用已确认的绝对命令映射。最终比较状态跟踪效果，而非仅按训练 action loss 排序。

若缺少接触/力反馈而残差主要集中在接触阶段，应记录可观测性限制，不能默认扩大 MLP 就能解决。关节力矩逆动力学不属于第一版；单个末端 delta 无法确定接触力或唯一关节解。

**7. 验证安排：把几何、命令和执行效果分开**

首个 300 转移片段的时序划分预先固定：

| 用途 | frame_index | 数量 |
| --- | --- | --- |
| fit | `[30,210)` | 180 |
| 隔离区 | `[210,220)` | 10 |
| validation：选择 lag / ridge / 特征规模及模型内控制增益 | `[220,260)` | 40 |
| 隔离区 | `[260,270)` | 10 |
| test：一次性报告 | `[270,330)` | 60 |

禁止打散相邻帧做随机划分。构造带历史/延迟的样本后，删除输入或标签跨越所属区间边界的行；上表是原始划分，实际有效计数另报。若扩大最大 lag、history 或训练样本的目标 horizon，隔离区同步扩大。初始数据留出非常小，只用于单轨迹可行性判断；后续泛化验证必须保留完整独立 episode。

PID 积分具有持续记忆，10 帧隔离区本身不能消除控制器状态泄漏。fit/validation/test 的控制 rollout 分别初始化积分和滤波状态，并使用各自区间内固定、明确的 warm-up；计分剔除 warm-up，不跨划分带入积分。实际部署跨 chunk 延续状态，与离线独立评估的初始化规则分开记录。

按以下四类证据分别报告：

实验矩阵覆盖：A 解析转换、A+线性前馈校准、相同前馈下的 P/PI/PD/PID、有界增益调度、小型直接逆模型及残差逆模型。每项记录数据、控制配置、计算量、适用输入与完成状态；不可运行的项记录具体缺失条件，不把未执行项计为通过。

| 验证问题 | 指标与基线 | 能说明什么 |
| --- | --- | --- |
| 几何定义是否正确 | 单步/累计 position error（m）、SO(3) geodesic error（rad）、TCP↔flange round trip、夹爪翻转；初始容差 `1e-5 m / 1e-4 rad` | 解析表示转换是否正确；不代表控制成功 |
| 是否接近记录的命令 | A/校准前馈与 R1 Cartesian 命令的逐轴 MAE、RMSE、NRMSE、动态段相关性；比较零增量/保持、上一条命令、方案 A；夹爪单独评分 | adapter 对该轨迹历史命令的重建能力，不能据此选择闭环 PID |
| 是否可能产生期望运动 | 独立留出数据上的前向预测误差、命令平滑度、饱和比例、状态/速度依赖、接触与自由运动分段 | 提供进一步闭环验证的依据；拟合的前向模型自洽不能当真机证据 |
| 反馈能否纠正偏差 | 相同参考、初态及扰动下比较前馈、前馈+P、PI、PD、PID；报告位置/姿态 RMSE、终点误差、静止目标稳态误差/超调/调节时间、噪声响应、饱和时长及控制变化率 | 区分模型内、独立仿真、真机三种证据；先测 oracle 参考，再测世界模型参考 |

NRMSE 使用 fit 段固定尺度；近常量轴单独标记，不让夹爪、无效槽或静止段主导均值。静止/运动分段阈值及接触标注方法在打开 test 结果前确定。IK 有冗余，因此 joint command error 只能作为辅助指标，不能要求所有合理动作都复现相同 joint target。

暂定进入扩展验证的信号：前馈校准在有激励的 arm 通道上相对 A/保持基线降低留出 command NRMSE；闭环候选则在预先固定的非零偏差/扰动场景上，相对相同前馈的无反馈版本将位置/姿态尺度平衡后的 tracking error 降低至少 20%，且主要通道、超调、饱和时间与控制变化率没有明显恶化。后者才是引入 PID 的主要判据；单项 command RMSE 下降不能代替。阈值是暂定实验标准，实际容许误差随已确认的控制接口在 test 前固定。没有改善时应报告是哪一层失败，而非直接堆网络。

后续闭环验证采用两阶段：先输入真实未来 delta 的 oracle 参考轨迹验证 adapter，再输入同一冻结世界模型产生的 delta。比较相同初态/任务下的跟踪误差、最终位姿、任务成功率及发送到反馈的延迟。仿真需要匹配控制接口及参数；仿真通过也不能替代真机结论。这些闭环工作在本次 plan 之后开展。

**8. 实施步骤与交付物**

| 顺序 | 工作 | 交付与完成条件 |
| --- | --- | --- |
| P0 | 固定数据、metadata、版本、内外环接口与时间合同；抽取上述 300 转移片段 | `sample_manifest.json`、`sample.npz`、`data_audit.json`；来源与时间/mask 检查通过，历史控制器及反馈可用性未确认项显式保留 |
| P1 | 实现方案 A，完成几何重建与候选控制编码 | `analytic_commands.npz`、`geometry_metrics.json`；几何检查通过；原始命令的接口代数一致性检查单独报告 |
| P2 | 辨识局部前向响应，校准前馈，建立有状态的外层 P/PI/PD/PID | `adapter_params.json`、`pid_config.json`、`calibration_metrics.json`；验证响应模型，核查物理单位、anti-windup、滤波、重置与重规划 |
| P3 | 在局部模型推进状态的 rollout 上比较各候选反馈，汇总证据 | `closed_loop_model_metrics.json`、`report.md`、逐轴/误差/积分/命令曲线；分别报告历史命令拟合与模型内扰动恢复 |
| P4 | 扩展同设备 episode，比较方案 C 与匹配接口的独立仿真结果 | episode 隔离验证；固定参数后测留出轨迹；增益调度、直接逆与残差网络均进入实验矩阵 |
| P5 | oracle / predicted delta 分开进行真机闭环验证，再考虑集成 | 确认内外环职责、频率与执行指标；在此之前不宣称模型内 PID 结果等同真机控制 |

独立仓库的目录约定（基础结构已建立，具体算法和实验文件按阶段添加）：

```text
s2a/
  README.md
  AGENTS.md
  pyproject.toml
  configs/droid_episode0.json
  plans/v0.0.1/001_state_to_action.md
  src/s2a/                     # 几何、数据对齐、逆模型、控制与评估模块
  scripts/                     # 实验入口；从配置读取路径
  tests/                       # 单元测试和独立集成测试
  reports/                     # 版本管理中的实验报告
  outputs/droid_episode0/       # 生成后忽略；保存数据子集、参数、指标和图
    sample_manifest.json
    sample.npz
    data_audit.json
    analytic_commands.npz
    adapter_params.json
    pid_config.json
    geometry_metrics.json
    calibration_metrics.json
    closed_loop_model_metrics.json
    figures/
```

后续命令均在 `/share/project/eai_pwm/home/hcr/repos/s2a` 执行，环境建立方式见 `README.md`。脚本显式设置 `--seed 42`；本机已有环境也可使用 `LD_LIBRARY_PATH='' /opt/venv/bin/python ...`。P0–P3 的数据分析与局部模型 rollout 可使用 NumPy / SciPy / PyArrow 在 CPU 完成，不等同高保真仿真。后续实现需验证 column-rot6d、SE(3) 左右乘、TCP 杠杆项、独立 gripper 模式、尾行 mask、物理 dt，以及积分饱和/释放、微分滤波、参考切换、陈旧反馈和 PID 状态隔离；本次项目基础结构不新增占位测试。

实现和测试统一留在本仓库；包括未来可复用的 runtime adapter。Cosmos 仓库及 R1/R2 发布数据只作外部只读参考，后续如需集成世界模型，由调用方通过 S2A 接口接入。S2A 的 `plans/`、`configs/` 和 `reports/` 纳入版本管理，`outputs/`、本地数据、权重和虚拟环境由根目录 `.gitignore` 忽略；不继承 Cosmos 忽略整个 plans 目录的规则。

**9. 第一轮必须回答的五个问题**

1. canonical TCP 的 observed delta 是否能精确还原下一状态，并正确转换到真实接口的 flange 坐标系？
2. 还原的下一状态与原始 commanded target 差多少，哪些部分可由前馈校准解释，哪些需要实时反馈纠偏？
3. 在相同参考与扰动下，前馈 + P/PI/PD/PID 各自改善什么，I/D 是否确有增益？
4. 当前数据能否支撑响应辨识和增益整定，闭环改进能否在独立留出轨迹与匹配接口的仿真中保持？
5. 现有数据还缺哪项设备或控制器信息，才能从离线可行性推进到该款真机的闭环验证？

本轮终点是拿出一个有证据的最小转换方案，或准确指出不能由这段数据识别的部分；不以训练出一个网络作为成功标准。
