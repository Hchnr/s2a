# H12 后续原生物理入口与运行时路径调查（2026-10-29）

本报告收敛已完成的只读源码、元数据、路径存在性及 SHA256 调查，供后续接续使用。没有建立新运行时，没有导入仿真、加载模型或执行物理周期，没有取得新的物理控制资格。已执行旧源码、配置、计划与结果保持不变；GPU 继续禁用。

**只有原 actual368 / H12 的完整模型 witness、全部响应独立核验及对应 owner 正常关闭均通过后，才可另行登记新的 CPU 物理 pilot。** 有限搜索下降、prefix 增长或部分 checker 日志均不满足这个前置。本报告不授权启动物理或扩大当前实验预算。[本轮计划:3](../plans/v0.0.1/prefix_optimization_reserve_20261029.md#L3)、[完整 witness 判定:163](../src/s2a/prefix_callback_backtracking_plan.py#L163)

## 原始来源与安全路径候选

原 manifest 为：

```text
outputs/tracking_preview_wam_development_v5_20261028/development/cases/task0_init0_s42_terminal058/simulation/8400_masked_position_hold/task0_init0_s42/manifest.json
SHA256 4a42c9647372bfce84f63bde93bb066d0ee06d7157cce9d8cef99271ee3539ab
```

本次实算 SHA 与原 registration 中引用一致。三个运行配置字段位于 [manifest:2057](../outputs/tracking_preview_wam_development_v5_20261028/development/cases/task0_init0_s42_terminal058/simulation/8400_masked_position_hold/task0_init0_s42/manifest.json#L2057)。它们的旧 `/share/project` 路径均已不存在；下表 `/share-mixed/project` 候选存在，仅作只读来源，不向旧路径写回或恢复。

| 字段 | 原路径 | 当前安全候选 |
| --- | --- | --- |
| `libero_root` | `/share/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_sim/LIBERO` | `/share-mixed/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_sim/LIBERO` |
| `libero_config` | `/share/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_sim/config/config.yaml` | `/share-mixed/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_sim/config/config.yaml` |
| `osmesa_library` | `/share/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_sim/systemlibs/usr/lib/x86_64-linux-gnu/libOSMesa.so.8.0.0` | `/share-mixed/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_sim/systemlibs/usr/lib/x86_64-linux-gnu/libOSMesa.so.8.0.0` |

针对原 manifest 已登记的实际运行源码，安全 LIBERO 副本的 **45/45** 个 Python 文件 SHA 匹配；当前仓库 `outputs/libero_env311/lib/python3.11/site-packages/robosuite/` 的 **120/120** 个已登记源码 SHA 亦匹配。历史清单见 [manifest 的 simulator_source_sha256:4138](../outputs/tracking_preview_wam_development_v5_20261028/development/cases/task0_init0_s42_terminal058/simulation/8400_masked_position_hold/task0_init0_s42/manifest.json#L4138)。这些计数只覆盖历史已登记源码，不表示整个环境和全部数据依赖已获资格。

两个安全文件的实算 SHA 均匹配已有历史登记：

| 文件 | 历史及当前 SHA256 | 历史依据 |
| --- | --- | --- |
| `config/config.yaml` | `d7ece7fb95f0fa9462b60b97632e2b7ece297f074d9c66433c0002018e9a114e` | [配置登记:91](../configs/wam_video_pose_restore_20261020.json#L91) |
| `libOSMesa.so.8.0.0` | `40e28f0f15c4cd13ad3c353509eb2fad79d8041528c59dc56d6e68fae8d1677a` | [库登记:95](../configs/wam_video_pose_restore_20261020.json#L95) |

仓库内 [旧 H12 pilot 的 config 副本](../outputs/tracking_v5_fk_first_long_h12_native_check_20261028/native/libero_config/config.yaml#L1)也具有上述 config SHA。尚未确认仓库内已有完整、已登记的 LIBERO 与 OSMesa 副本；外部安全副本可作为后续仓库内快照的只读来源。

## 最小路径迁移与未闭合项

安全 `config.yaml` 与旧文件字节相同，因此其内部 **五个值仍全部指向旧根**：

| 配置键 | 旧路径（共同前缀为 `/share/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_sim/`） |
| --- | --- |
| `benchmark_root` | `LIBERO/libero/libero` |
| `bddl_files` | `LIBERO/libero/libero/bddl_files` |
| `init_states` | `LIBERO/libero/libero/init_files` |
| `assets` | `LIBERO/libero/libero/assets` |
| `datasets` | `data` |

仅替换外层 `libero_config` 路径不能解决这些引用。后续最小迁移需保留原 manifest/config 及 SHA，新增仓库内派生运行配置，显式记录三个外层字段和五个内层值的路径映射、原始与新 SHA，并核对确实消费的代码/数据。不能更改 seed、任务、控制频率、恢复策略或其他实验参数。现有入口会直接读取 config、加入 LIBERO 路径并加载 OSMesa，见 [configure_simulator:27](../src/s2a/libero_simulation.py#L27)。

尚有两个明确缺口，本次未扩展执行检查：

1. **Bootstrap 环境依赖未闭合。** `make_environment` 先从 BDDL 构建 `OffScreenRenderEnv` 并 reset，然后才加载记录 XML。因此对应任务 BDDL、初始化模型 XML 及其额外资产也必须可用并绑定来源。当前96资产快照只证明最终完整数值模型的资产闭包，不证明 bootstrap 所需模型闭包。[环境构建:72](../src/s2a/libero_simulation.py#L72)、[最终模型快照验证:142](../src/s2a/prefix_trust_asset_repair.py#L142)
2. **OSMesa 传递库依赖未闭合。** 已核对目标 `.so` 的历史字节身份，尚未验证它在新路径下的全部动态依赖与实际加载。不能把文件存在和 SHA 匹配写成运行时加载成功。

## Task0 bootstrap 输入的有界补查

本次补查仅沿原 task0 的构建入口读取源码、JSON 和对应 BDDL，未构建环境或遍历数据集。历史引用核对范围为原 manifest、`outputs/wam_checkpoint_acceptance_20261019v2/samples/manifest.json`、`outputs/terminal_development_20261025/historical_oracle_xml_migration.json` 及仓库 configs。下文“未找到历史绑定”限于这个已检查范围。

**本实验的 task0 是 `libero_goal` 任务8，即 bowl/plate。** 原 runtime 登记 `task_suite="libero_goal"`、`task_id=8`；`make_environment` 据此取任务，并用 `get_libero_path("bddl_files") / task.problem_folder / task.bddl_file` 构造首次环境。安全 LIBERO 副本中 `libero/libero/benchmark/__init__.py:84,111,188` 的默认 task order 为 identity，`libero/libero/benchmark/libero_suite_task_map.py:35` 的索引8是 `put_the_bowl_on_the_plate`；这两份源码与原 manifest 登记 SHA 一致。[原任务参数:2074](../outputs/tracking_preview_wam_development_v5_20261028/development/cases/task0_init0_s42_terminal058/simulation/8400_masked_position_hold/task0_init0_s42/manifest.json#L2074)、[构建入口:74](../src/s2a/libero_simulation.py#L74)

对应 BDDL 安全候选及本次实算指纹为：

```text
/share-mixed/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_sim/LIBERO/libero/libero/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl
SHA256 358b7781437e412c67d6a7648c1c28796b6b1bf117a25e61316cb8dfeb940e5b
```

这是**当前文件指纹，尚缺历史字节身份绑定**。不能将它追认为原运行时已验证 SHA。[已有 restore 配置的 task_bddl:99](../configs/wam_video_pose_restore_20261020.json#L99)实际是 `put_the_wine_bottle_on_the_rack.bddl`，不能用于证明本任务的 bowl/plate BDDL。

**恢复初态已有精确历史绑定，不依赖 `.pruned_init`。** 构建入口直接消费原 manifest `record.initial_state` 内的79个 float64，并在恢复前验证其字节 SHA；本次按小端 float64 重算得到 `5239d6a1c021e69231f3ef91a1ad93bd5b626981c003140b29150c0c9713a702`，与原登记完全一致。`get_task_init_states()` 是 LIBERO 的另一个接口，本入口没有调用；也不读取 runtime 的 episodes fixture、原 HDF5 或 expert.npy。这些来源可保留 provenance，但不能误列为此固定命令 pilot 的直接 bootstrap 输入。[初态校验与恢复:97](../src/s2a/libero_simulation.py#L97)、[原初态 SHA:2241](../outputs/tracking_preview_wam_development_v5_20261028/development/cases/task0_init0_s42_terminal058/simulation/8400_masked_position_hold/task0_init0_s42/manifest.json#L2241)

原 manifest 的整体 SHA 同时绑定 runtime 参数：seed42、libero_goal任务8、256渲染分辨率、`camera_observations=False` 和 `refresh_controller_after_restore=True`。入口将 seed+trial 用于 random、NumPy 和环境，并在恢复后刷新 controller。需要保留一个默认值边界：它不把 runtime 的 `control_fps/max_steps` 传入构造器；实际20Hz和 horizon1000来自已登记 `libero/libero/envs/env_wrapper.py:27` 的默认参数，而非这两个 runtime 字段。[构建与种子:75](../src/s2a/libero_simulation.py#L75)、[恢复后刷新:105](../src/s2a/libero_simulation.py#L105)、[原 runtime:2091](../outputs/tracking_preview_wam_development_v5_20261028/development/cases/task0_init0_s42_terminal058/simulation/8400_masked_position_hold/task0_init0_s42/manifest.json#L2091)

首次构建还会读取**默认 OSC JSON**：`env_wrapper.py:47` 请求 `OSC_POSE`，robosuite 根据名称加载 `controllers/config/osc_pose.json`。当前仓库文件 SHA 为 `07f3a8b41d3eecc8a36700a85bd8207ee3fbc36eb73f6a2248340c96770d1e45`，同样尚未在上述历史引用范围找到字节身份绑定。其当前值包括 kp150、阻尼比1、输入±1、输出平移±.05/旋转±.5、fixed impedance、uncoupled/control_delta开启及无 interpolation；源码 SHA 匹配并不自动绑定这个外部 JSON。[加载路径:76](../outputs/libero_env311/lib/python3.11/site-packages/robosuite/controllers/controller_factory.py#L76)、[当前配置:1](../outputs/libero_env311/lib/python3.11/site-packages/robosuite/controllers/config/osc_pose.json#L1)

此次对相关四份源码再次实算，均匹配原 manifest 的历史登记：

| 源码（LIBERO 路径相对于安全 LIBERO 根） | 已登记且当前匹配的 SHA256 |
| --- | --- |
| `libero/libero/benchmark/__init__.py` | `4ce6edbb00b3280692f2e2b2e8a14bf6e2d5878965b0484a386c0355a867ecc5` |
| `libero/libero/benchmark/libero_suite_task_map.py` | `0c950df0a785aa55de968bb38ccd865d2017f71ddbe6f48cfd05ac0742b6d62d` |
| `libero/libero/envs/env_wrapper.py` | `a782fb76c9792268d28979474fe72849e1e98ada49c8e997a65359a8d6b6acd0` |
| 仓库 `outputs/libero_env311/lib/python3.11/site-packages/robosuite/controllers/controller_factory.py` | `1dab09602c41dcf3749bed43dd3bc4e098040cabc8f7fbfd030d9008bd3d5776` |

历史依据分别见 [任务选择源码:4140](../outputs/tracking_preview_wam_development_v5_20261028/development/cases/task0_init0_s42_terminal058/simulation/8400_masked_position_hold/task0_init0_s42/manifest.json#L4140)、[环境包装源码:4154](../outputs/tracking_preview_wam_development_v5_20261028/development/cases/task0_init0_s42_terminal058/simulation/8400_masked_position_hold/task0_init0_s42/manifest.json#L4154)、[controller factory:4187](../outputs/tracking_preview_wam_development_v5_20261028/development/cases/task0_init0_s42_terminal058/simulation/8400_masked_position_hold/task0_init0_s42/manifest.json#L4187)。原 tool contract 文件亦匹配历史登记 `6fcaed3fe499aacb15d03170d002b1c897091481a0022db5f2922c06320a3a0c`，但它不是默认 OSC JSON 的替代证明。[tool contract 登记:6](../outputs/tracking_preview_wam_development_v5_20261028/development/cases/task0_init0_s42_terminal058/simulation/8400_masked_position_hold/task0_init0_s42/manifest.json#L6)

因此，新增明确缺口为本任务 BDDL 和默认 OSC JSON 的历史身份；bootstrap XML及资产闭包也仍未核验。首次 BDDL 构建和 reset 先于记录 XML 恢复，已有最终模型96资产快照不能覆盖或免除这一步。以上仅是只读输入调查，没有取得新的运行时或物理资格。

## 完整 witness 后的最小物理接入建议

采用新的独占 producer/checker/owner 输出，不修改已执行旧入口。可复用 `probe_tracking_v5_fk_first_long_native.py` 的原初态、seed42、368条历史命令重放及观测调度，从自然执行恢复 actual368 的控制器、夹爪和接触求解历史；逐端点核对原历史至368。[原前缀重放:117](../scripts/probe_tracking_v5_fk_first_long_native.py#L117)

在368显式核对新 witness 的 q/v、completed367 previous command、旋转 latch、`initial_joint`、固定手指状态和控制合同。`endpoint()`已有夹爪 motor cache 及控制器缓存记录；`read_current_dynamics()`可以在已刷新端点读取动力学而不执行额外 forward。[缓存采样:114](../scripts/collect_wam_native_training.py#L114)、[无状态推进的当前读取:28](../src/s2a/current_coupled_state.py#L28)

本次最小 pilot 应固定执行已独立核验的 `selected.commands[0:12]`，每周期25×2ms，原参考 cursor0..11每50ms消费一次，夹爪保持原367命令。替换旧入口的 first-pose-only 重规划段；该旧策略不能承接新 H12 全部 caps 的资格。[旧重规划段:195](../scripts/probe_tracking_v5_fk_first_long_native.py#L195)、[原九组约束:57](../src/s2a/joint_preview_constraint_audit.py#L57)

旧 actual369+不得进入新控制输入或作为仿真恢复目标。历史 loader 只向分派器暴露0..368，目标仅来自已生成 chunk46。执行后产生的新分支369..380观测可用于当步停止判断、实际计分及独立重放比较；不得拿旧分支未来替换这些真实结果。该最小方案不在中途新增优化、hold、跳点或参考时间重参数。

保持初始 refresh、每步 `endpoint()`、每8步 render、动作后 refresh 的原顺序；额外 `sim.forward()`可能影响后续接触动力学。[独立 checker 的观测协议:123](../scripts/check_tracking_v5_fk_first_long_native.py#L123)每个新增周期保存25个实际 q/qvel/time、命令、实际 TCP 误差、关节裕量与停止端点。沿用实际质量/裕量违规后停止下一次派发的语义，并在执行前登记实际 braking/terminal 审计的范围与资格；不得用数值比较容差扩张实际 pose caps。[实际采样与停止:259](../scripts/probe_tracking_v5_fk_first_long_native.py#L259)

接触、原始/截断力矩及约束力可复用只读动态快照，不增加 forward；原 pilot 未定义通用碰撞停止门槛，任何新增接触门槛须另行执行前登记。模型固定手指且不求解接触，因此完整模型 witness 不证明完整机械安全或抓取成功；对象和任务谓词只作评分记录。[动态快照:142](../scripts/replay_tracking_v5_velocity_jump.py#L142)、[数值模型边界:8](../src/s2a/mujoco_robot_dynamics.py#L8)、[原生任务与抓取记录:29](../src/s2a/checkpoint_simulation.py#L29)

完整预算应为：producer 最多 **368+12=380控制周期、381端点、300个新增内环样本**；独立 checker 按保存命令从同一初态重放，最多同量。两侧合计最多 **760控制周期**，不能只计12个新动作而漏掉历史前缀与独立重放。提前停止时按实际派发量和失败端点记账。独立核验可复用旧 checker 的全部状态/内步、命令、参考顺序、夹爪与停止后无动作检查，保持明确的数值容差及 owner 实际 wait/reap/source closure。[重放循环:138](../scripts/check_tracking_v5_fk_first_long_native.py#L138)、[停止检查:284](../scripts/check_tracking_v5_fk_first_long_native.py#L284)

## 证据边界

即使这12周期实际全部达标，也仅取得同一起点短段跟踪证据。完整控制仍须原 **18/18正确参考**、每条位置 RMSE≤.01m / SO(3) RMSE≤.05rad及全量独立核验；随后是原 **12条 WAM 开发矩阵**，最终仍为 **24条留出总体至少20/24稳定成功且每任务至少4/8**。GPU禁用期间不启动依赖模型的阶段。[正确参考门槛:11](../plans/v0.0.1/221_preview_physical_execution_20261026.md#L11)、[WAM开发门槛:5](../plans/v0.0.1/tracking_wam_v5_20261028.md#L5)、[最终验收:23](../plans/v0.0.1/036_project_goal_20261023.md#L23)

本报告中的45/45、120/120和文件 SHA 是只读比对结果，未生成或宣称新的运行时加载、仿真或控制资格收据。Bootstrap、传递库与后续物理协议均保留为未完成项。
