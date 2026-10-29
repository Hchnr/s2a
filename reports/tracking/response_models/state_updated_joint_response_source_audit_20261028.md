# 冻结/状态更新关节响应：原生控制器源码合同审计

2026-10-28；CPU 只读源码与模型参数审计。目标是固定下一数值对照的控制方程、积分含义和遗漏项，不是新的控制或任务资格。GPU、物理步、世界模型请求和 fresh 案例均为 0。本次只编译归档 XML 读取参数，没有创建 LIBERO 环境、调用 `mj_forward` 或 `mj_step`。

本报告所有相对路径以 `/share-mixed/project/eai_pwm/home/hcr/repos/s2a` 为根。下列缩写只是为了缩短源码定位：

```text
RS = outputs/libero_env311/lib/python3.11/site-packages/robosuite
MJ = outputs/libero_env311/lib/python3.11/site-packages/mujoco/include/mujoco
NP = outputs/libero_env311/lib/python3.11/site-packages/numpy
NB = outputs/libero_env311/lib/python3.11/site-packages/numba
CASE = outputs/tracking_preview_wam_development_v5_20261028/development/cases/task0_init0_s42_terminal058/simulation/8400_masked_position_hold/task0_init0_s42
```

## 来源与指纹

使用既有 H8 原生分支的来源链：`outputs/tracking_v5_fk_first_long_h8_native_20261028/registration.json:2` 指向同起点数值结果，`:14` 指向原生 manifest；`outputs/tracking_v5_fk_first_long_numeric_20261028/registration.json:26`、`:34` 固定 controller 和 XML。当前文件重新计算 SHA256 后，三项均与注册指纹相同。

| 文件 | SHA256 |
| --- | --- |
| `CASE/manifest.json` | `4a42c9647372bfce84f63bde93bb066d0ee06d7157cce9d8cef99271ee3539ab` |
| `CASE/controller.json` | `d6d143e51764c3fc496006315a6e1e51a9e2aa1314e5294acdae238a987ac2c9` |
| `CASE/model.xml` | `50d3ece2777882c860513208d8c5ac75a31f49ff9db31d034211b5879c98f0e6` |

实际 robosuite 源码位于上述环境的普通安装目录，不需要修改外部 Cosmos 仓库或依赖 editable 路径。以下 8 个当前源码 SHA256 均逐项匹配 manifest 的 `execution_sha256`（`:4`）及 `simulator_source_sha256`（`:4138`）。表中最后两列给出两个来源表的实际行号；它们也是本次匹配检查的可复核输入。

| RS 下源文件 | 当前与归档一致的 SHA256 | execution 行 | simulator 行 |
| --- | --- | --- | --- |
| `controllers/base_controller.py` | `61f99a2daeb4eb6e39bfb996d94c5a5689ef3bc771d030d070193ea56d6a8332` | 1786 | 4186 |
| `controllers/osc.py` | `cfa62a0e719bc53ef0c701efa66f7b3e2272d4fca2150ec73c05bb145eba85cb` | 1794 | 4194 |
| `environments/base.py` | `60eab4afd3dc95b4ac4cf0659fb9afe6360161912a4ac6931efe86ce1e1de519` | 1796 | 4196 |
| `macros.py` | `85041caa914fbee7c9976e610ef8e0e76036af796ea65f07abb45c57962c6b25` | 1813 | 4213 |
| `robots/robot.py` | `7a3a98eb75b4e7cb1b822a918d0c693d441356279a8e796cfc02eb1fd5963f23` | 1888 | 4288 |
| `robots/single_arm.py` | `38237d7b68c876b413f0df811b135ca3b2fb91061c011696baab07db5e87fa14` | 1889 | 4289 |
| `utils/binding_utils.py` | `5ac3c9da0cb9f85d694c6147103268df65151f751612a6c575ae0da4eb4988e4` | 1891 | 4291 |
| `utils/control_utils.py` | `1d0f1a063894b1e8289339e223665e14e8aec492c31e9e00f42edea8dc735f43` | 1893 | 4293 |

这些检查针对控制方程及调用链的关键文件，不等于重新核验完整环境闭包。NumPy、Numba 和 MuJoCo 头文件用于明确当前实现语义；未把它们混称为上述 8 项 SHA 比较的一部分。

## 固定控制接口与目标更新

`CASE/controller.json:3` 起记录归一化 arm 输入 `[-1,1]^6`，物理输出前三维 `[-.05,.05] m`、后三维 `[-.5,.5] rad`，六维 `kp=150`、`kd=24.49489742783178`，20 Hz 与 2 ms。控制 SITE 为 `gripper0_grip_site`，坐标系为 MuJoCo world；TCP 转换仍是独立工具合同，不能把 SITE 位置误差直接写成 TCP 误差。

| 行为 | 原生依据 |
| --- | --- |
| 输入先按 input min/max clip，再 affine scale 到物理 delta | `RS/controllers/base_controller.py:104`、`:116` |
| 外环首个内环才调用 `set_goal`；其他内环保持同一 goal | `RS/robots/single_arm.py:245`；`RS/environments/base.py:387`、`:396` |
| 位置 goal 为当前控制器 SITE 位置加 world delta | `RS/controllers/osc.py:264`；`RS/utils/control_utils.py:132` |
| 非零旋转 goal 为 `Exp(delta_rot) @ R_current`，左乘 world 轴角增量 | `RS/controllers/osc.py:257`；`RS/utils/control_utils.py:174` |
| 三个旋转分量全零时保留已有方向 goal，不自动把它重置为当前方向 | `RS/controllers/osc.py:259` |
| `reset_goal` 取控制器缓存的当前位置/方向，本身不先调用 `update` | `RS/controllers/osc.py:366` |
| `initial_joint` 是原生控制器保留的 nullspace 目标，不是每个预测周期的当前 q | `RS/controllers/base_controller.py:89`、`:175`；`RS/controllers/osc.py:350` |
| 本地环境合同拒绝有位置或方向插值器的控制器 | `src/s2a/libero_simulation.py:203`、`:207` |

默认 OSC 配置也明确 fixed impedance、uncoupled、delta、无插值和无 goal limits，见 `RS/controllers/config/osc_pose.json:9`。实际原生查询再次要求 uncoupled fixed delta 且无 goal limits，见 `src/s2a/current_coupled_state.py:38`。新数值入口只覆盖非零旋转 reset；遇到全零旋转应拒绝，或者显式传入历史方向 goal 后另行定义合同。

## OSC 方程、补偿与限幅

令 `J=[J_pos;J_ori]` 为 world SITE Jacobian，`M` 为七 arm DOF 的完整惯性子块。原生 `orientation_error` 的 docstring 将它描述为轴角，但实际代码是

```text
e_rot = 0.5 * Σ_i cross(R_current[:, i], R_goal[:, i])
e = [p_goal - p_current, e_rot]
w = Kp * e - Kd * (J v)
```

应跟随实际实现的 sine-angle 误差，而不能替换为 SO(3) log。依据：`RS/utils/control_utils.py:102`；`RS/controllers/osc.py:314`。评价 TCP 姿态误差仍可使用项目既有 SO(3) log，须明确它与控制器内部角误差不同。

```text
Lambda_full = pinv(J M^-1 J.T)
Lambda_pos = pinv(J_pos M^-1 J_pos.T)
Lambda_ori = pinv(J_ori M^-1 J_ori.T)
N = I - M^-1 J.T Lambda_full J
tau_task = J_pos.T Lambda_pos w_pos + J_ori.T Lambda_ori w_rot
tau_null = N.T M [10 (q_initial - q) - 2 sqrt(10) v]
tau_requested = tau_task + tau_null + qfrc_bias
tau_applied = clip(tau_requested, arm_ctrlrange)
```

原生分别计算位置/方向 Lambda，完整 Lambda 用于 nullspace；`uncoupling` 分支见 `RS/controllers/osc.py:330`。矩阵定义见 `RS/utils/control_utils.py:62`，nullspace 增益和力矩顺序见 `:8`、`:32`。`pinv` 不设置定制阈值：NumPy 默认 `rcond=1e-15`，仅保留 `s > rcond*smax`，见 `NP/linalg/linalg.py:1937`、`:2025`；Numba 对应默认及严格比较见 `NB/np/linalg.py:1966`、`:2070`。

`torque_compensation` 返回整个 `qfrc_bias`，不只是重力项，也不包含 `qfrc_passive`，见 `RS/controllers/base_controller.py:230`；MuJoCo 对该数组标注为 `C(qpos,qvel)`，见 `MJ/mjdata.h:296`。OSC 输出相加顺序见 `RS/controllers/osc.py:344`。真正的 actuator clip 在 `RS/robots/single_arm.py:249`，先包含 bias/nullspace 后统一限幅，再写入 ctrl；限幅来源为 `actuator_ctrlrange`，见 `RS/robots/robot.py:273`。不能先 clip 无 bias 力矩再加 bias。

## 缓存更新与积分设置

原生不是整 50 ms 冻结 M/J：`RS/controllers/base_controller.py:125` 在 `new_update=True` 或 `force=True` 时调用 `sim.forward`，重新取得 SITE pose/速度、joint q/v、J 和完整 M 的 arm 子块；`:160` 清除标志。`run_controller` 收尾在 `:95` 将 `new_update=True` 留给下一个内环。这样外环首内环的 `set_goal` 和随后的 `run_controller` 可以共用同一次刷新，后续每个内环重新刷新。

`RS/environments/base.py:391` 每个控制周期运行 `int(control_timestep/model_timestep)` 次，顺序为 `sim.forward → _pre_action → sim.step → observables`。`RS/macros.py:11` 为 `.002 s`，20 Hz 对应 25 次。`RS/utils/binding_utils.py:1086`、`:1090` 分别直接调用 `mj_forward`、`mj_step`。已有 H8 内环观察只在原生 step 后读取 q/v/time，未增添 forward，见 `scripts/probe_tracking_v5_fk_first_long_native.py:127`；本次审计不改变该历史证据。

本次在 `CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=disable` 环境下，只用原生 Python 加载 MuJoCo 并 `MjModel.from_xml_path` 编译归档 XML，读取到：

| 参数 | 编译值及源码/XML 依据 |
| --- | --- |
| MuJoCo / integrator | `2.3.7` / `0 = mjINT_EULER`；`MJ/mjmodel.h:128` 标注 semi-implicit Euler |
| timestep / disableflags | `.002 s` / `0`；XML 未覆写 timestep 或 integrator，`CASE/model.xml:3` |
| gravity / wind | `[0,0,-9.81]` / `[0,0,0]` |
| density / viscosity | `1.2` / `2e-5`，`CASE/model.xml:3` |
| arm damping | `[.1,.1,.1,.1,.1,.01,.01]`，七 joint 定义自 `CASE/model.xml:238` 起 |
| arm frictionloss | 七维均 `.1`，同一组 joint 定义 |
| arm stiffness | 七维均 `0` |
| armature | `[5,2.5,1.66667,1.25,1,.833333,.714286]`，同一组 joint 定义 |
| arm actuator 类型 | `dyntype=0`、`gaintype=0`、`biastype=0`；gear 均 `[1,0,0,0,0,0]`，gain 第一个参数为 1、其余为 0，bias 参数全 0 |
| arm actuator limits | 前五关节 `[-80,80]`，最后两关节 `[-12,12]`；`ctrllimited=True`、`forcelimited=False`，`CASE/model.xml:581` |
| fingers | armature 1、damping 100、frictionloss 1，`CASE/model.xml:319`、`:329`；位置 actuator 和力限幅定义见 `:588` |

编译输出是本次交互中的只读审计观察，未新建实验产物目录，不将其冒充已独立审核关闭的实验结果。原模型质量矩阵包含 armature；使用 `qM` 时不应再次添加同一对角惯量。

## 数值对照边界与待验证项

保留旧 frozen analytic 响应作为第一路；新增两路数值模型使用相同 FK、非线性内部角误差、held goal、bias/passive 更新、统一 torque clip 和积分器。只改变 M/J 是否随预测 q 更新，且明确这个选择同时作用于 task inertia、nullspace、力矩映射、控制器阻尼 `Jv` 和 free plant M。两数值模型都沿各自预测状态更新 FK、bias 和被动力；它们的未来 q/v 不同，不能要求这些量跨路径数值相同。

可用的无 contact 查询顺序为 `mj_kinematics → mj_comPos → mj_crb → mj_comVel → mj_rne(flg_acc=0)`，再取 `mj_fullM`、`mj_jacSite`；依据 `MJ/mujoco.h:298`、`:313`、`:325`、`:334` 及 `MJ/mjdata.h:274`。这条路径不需要推进时间、建立约束或使用未来对象状态。`mj_passive` 可求 spring/damping/fluid，见 `MJ/mujoco.h:328`；其 dof damping 部分应与隐式阻尼积分分开，避免重复计入。

拟采用的数值积分合同为

```text
passive_extra = qfrc_passive + D v
(M + dt D) v_next = M v + dt (tau_applied - bias + passive_extra)
q_next = q + dt v_next
```

这是对角 joint damping 隐式处理的 semi-implicit Euler 自由模型，不能称为完整 MuJoCo 步进复制。明确保留以下限制：

- 无 contact、joint-limit force、joint frictionloss 约束；`mj_passive` 包含流体被动力，但不包含这些约束力。
- 只保留 arm M 子块，遗漏动态 gripper 加速度耦合；finger q 和 qvel 固定在该周期起点只是当前条件近似，不能声称它们构成自洽的未来积分轨迹。
- 所有外部对象只采用 XML 默认值，不能接收实际或未来对象状态作为预测输入；真实对象信息只能用于事后任务评价。
- 低层 passive 通路若推广到有 tendon 的 XML，需要单独准备 tendon 长度/速度等前置量。当前归档 XML 没有 tendon 定义；本次审计未证明任意 XML 都适用同一精简调用序列。
- 旧 analytic 路与新数值路还存在非线性角误差、限幅、被动力和离散积分差异；只有新增两数值路之间才能把变化归于冻结/更新 M/J 的控制选择。

九个既有真实周期应各从该周期起点 measured robot/finger 状态初始化一次，只沿预测 q/v 推进 25 次；actual inner q/v 与 endpoint TCP 只能事后评价。执行前仍需核对起点 M/J 与归档一致、低层查询与独立参考一致、nullspace/角误差/限幅与隐式 damping 没有实现错误。源码审计通过不替代这些数值验证，也不恢复 GPU、控制资格或最终 fresh 验收。
