# S2A — State to Action

S2A 把世界模型给出的期望 state/delta_state 转成固定控制器可接收的 action，使执行后的实际状态接近目标。

2026-10-20 独立物理评测正在执行：四份 checkpoint 统一 H40/执行8，另在旧集合固定执行16/40对照。新集合 native 和 oracle＋冻结S2A 均已独立复核24/24稳定成功；WAM已有28条完整轨迹，2条曾成功、1条稳定成功。将所有未完成项都假设成功，四份checkpoint也已无法达到本轮预设稳定性门槛；完整多任务结果和主要失败原因仍待后续矩阵及窗口对照。该集合与旧物理评测分离，但包含训练catalog来源及1条旧离线样本重叠。当前协议、来源边界和运行进度见[实施记录](plans/v0.0.1/record_20261020.md)与[阶段结论](plans/v0.0.1/conclusion_20261020.md)。

```text
已生成的期望 delta chunk → 带时间的 SE(3) 参考轨迹
                                  ↓
当前实测状态 + 因果历史 → S2A → 固定控制器 → 机器人/环境 → 新实测状态
```

主要评价 `F_C(s_now, G_C(...))` 的状态跟踪误差；历史 action 误差仅作诊断。必须区分 measured state、desired state、observed delta 和 controller command。

2026-10-19 v2四层验收已完成：2688次离线预测、90条物理仿真/51480步。确认并在S2A评测服务修正了推理无效Action128槽加噪与训练置零的差异；最新8800在原生H40窗口下曾成功1/12、末端成功1/12，成功任务为bowl→plate，约5秒成功后保持至30秒结束。H16下8400为2/12曾成功、1/12末端，8800为0/12；正确state经冻结S2A为12/12。完整结果见[四层验收报告](reports/wam_checkpoint_acceptance_20261019v2.md)、[state报告](reports/wam_state_acceptance_20261019v2.md)和[最终结论](plans/v0.0.1/conclusion_20261019v2.md)。这是固定WAM/固定S2A、不计推理延迟的独立仿真证据；修正尚未合入外部Cosmos生产代码，未执行真机实验。

已实现 DROID 单 episode 的只读抽取、几何转换、参考接口编码、局部响应辨识、在线有状态 PID adapter 和模型内扰动对照。初轮使用真实未来状态形成 oracle 参考。后续新增 LIBERO 独立 MuJoCo 仿真和固定 Cosmos checkpoint 的成对评测，执行状态见 [仿真报告](reports/libero_simulation_20261016.md)。完整设计见 [实施计划](plans/v0.0.1/001_state_to_action.md)，过程见 [实施记录](plans/v0.0.1/record_20261016.md)，完成状态及限制见 [结论](plans/v0.0.1/conclusion_20261016.md)。

2026-10-19 扩展修复了恢复仿真状态后的OSC缓存同步，区分HDF5动作后观测与当前R1/R2发布观测，并加入开度方向保持、释放锁存、输入延迟/噪声及终态保持对照。该阶段状态见 [控制审计报告](reports/libero_control_audit_20261019.md)、[本轮记录](plans/v0.0.1/record_20261019.md) 和 [本轮结论](plans/v0.0.1/conclusion_20261019.md)。候选的实验收益与部署资格分别判断，所有结果均无真机执行。

继续实施新增catalog所指当前发布帧的配对预测诊断，以及观测时间驱动的异步参考接入。`TimedReference`保留采集时的world锚点和时间轴；`AsyncReferenceBuffer`拒绝过期/乱序结果、跳过已过去的endpoint，空窗固定保持。新仿真逐步刷新当前端点反馈，只在需要请求模型时采集相机；物理控制与HTTP推理并行，实际墙钟超时单列记录。接口可执行、目标正确、任务成功和实时部署分别验证。

入口配置为 `configs/libero_followup_20261019.json`。依赖此前冻结的来源bridge和基线初始图像/预测，路径与指纹随结果保存；复跑请另设output_dir，已有结果不覆盖。以下命令均从本仓库执行，模拟器和模型服务仍是可选环境：

```bash
LD_LIBRARY_PATH='' .venv/bin/python scripts/prepare_libero_followup.py --seed 42
LD_LIBRARY_PATH='' .venv/bin/python scripts/audit_libero_alignment.py --seed 42
LD_LIBRARY_PATH='' .venv/bin/python scripts/audit_libero_release_revision.py --seed 42
# 独立终端运行本轮自有模型服务；结束后只停止其记录的进程组
LD_LIBRARY_PATH='' .venv/bin/python scripts/serve_libero_policy.py --config outputs/libero_followup_20261019/serving_config.json --seed 42 --gpu 0 --port 8380
LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/probe_libero_training_images.py --seed 42
LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/verify_libero_camera_capture.py --seed 42
# 编排器从显式simulation_python启动物理worker；live串行避免服务排队混淆延迟
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_libero_async.py --seed 42 --stage synthetic
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_libero_async.py --seed 42 --stage live
LD_LIBRARY_PATH='' .venv/bin/python scripts/summarize_libero_followup.py --seed 42
LD_LIBRARY_PATH='' .venv/bin/python scripts/plot_libero_followup.py --seed 42
```

WAM 效果专项已完成55项完整物理实验/50739步：不计推理延迟，两种纯WAM转换均0/5，共享解码正控均5/5；独立几何证据确认当前生成参考与任务不符，不能只归因为缺少转换。详见 [效果报告](reports/libero_wam_effect_20261019.md)。本专项暂停物理等待推理、不使用参考过期判断，用于判断效果。`configs/libero_wam_effect_20261019.json` 冻结共享 Action128 解码的 oracle/真实 WAM 对照，`configs/libero_wam_components_20261019.json` 冻结有特权的分量干预。完整请求、原始/有效参考、物理时钟、状态摘要与实际发送命令均保留。复现入口见 [scripts](scripts/README.md)，本专项运行结果追加于本轮记录及结论。

后续四层验收已确认原服务会向118个无效Action128维度注入噪声，与训练的置零合同不一致。上述历史结果描述原服务组合，不能单独用来否定冻结权重的任务能力；S2A评测服务中的mask修正及重新验收见 [v2记录](plans/v0.0.1/record_20261019v2.md) 和 [v2结论](plans/v0.0.1/conclusion_20261019v2.md)。外部Cosmos仓库保持只读，尚未将修正合入其生产服务。

```text
src/s2a/geometry.py        column-rot6d、SE(3)、TCP/flange、SO(3) 误差
src/s2a/data.py            provenance join、物理时钟、masks、样本与指纹
src/s2a/interface.py       固定版本 DROID 参考接口与独立夹爪 position
src/s2a/trajectory.py      固定锚点参考、插值、chunk 衔接
src/s2a/control.py         P/PI/PD/PID、状态、滤波、限幅和 anti-windup
src/s2a/runtime.py         StateToActionAdapter（生成命令，无硬件 I/O）
src/s2a/identification.py  ridge 前向响应与正则逆
src/s2a/study.py           时序选择、日志留出评估
src/s2a/rollout.py         候选命令推进新状态的局部模型对照
src/s2a/reporting.py       Markdown、JSON 摘要及 Matplotlib 曲线
scripts/                  实验入口
configs/                  数据路径、划分、所有网格及扰动参数
plans/、reports/           Git 内维护的设计、记录及精简结果
outputs/                  数据子集、参数、完整指标及图（Git 忽略）
```

Python 3.11+，从本仓库执行：

```bash
LD_LIBRARY_PATH='' python3 -m venv .venv
LD_LIBRARY_PATH='' .venv/bin/python -m pip install -e '.[dev,report]'
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
LD_LIBRARY_PATH='' .venv/bin/python -m pytest
LD_LIBRARY_PATH='' .venv/bin/python -m build --no-isolation --wheel --outdir outputs/dist
```

当前 NGC 主机也可用 `/opt/venv/bin/python -m venv --system-site-packages .venv` 继承系统 runtime 包；只向本仓库 venv 安装开发依赖，不修改共享环境。实际运行版本保存于 manifest，版本快照见 [constraints](configs/constraints_20261016.txt)；它不是跨平台完整锁文件。基础依赖只有 NumPy/SciPy/PyArrow，绘图需 `report` extra。

首个样本由 [配置](configs/droid_episode0.json) 固定为 DROID / Franka Panda + Robotiq 2F85 episode 0，转移 `[30,330)`，保留 endpoint 和 3 帧历史。数据根和明确的 shard 路径均由配置传入；S2A 不依赖 Cosmos 训练 dataloader、工作目录或模型权重。

```bash
# 完整 P0–P3：只读数据、参数选择、模型内 rollout、报告与曲线
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_experiment.py --config configs/droid_episode0.json --seed 42 --stage all --plots

# 可单独运行 extract / analytic / calibrate；报告重绘不重新拟合或打开 test
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_experiment.py --seed 42 --stage report --plots

# 产物/实验源码指纹与 70 个 test rollout 最终命令约束（不重新拟合）
LD_LIBRARY_PATH='' .venv/bin/python scripts/verify_artifacts.py --seed 42

# 显式读取外部发布数据的集成测试
LD_LIBRARY_PATH='' .venv/bin/python -m pytest tests -m integration --data-config configs/droid_episode0.json
```

默认测试不读取外部数据，集成测试标为 deselected，不算通过。显式运行时，文件缺失或契约不符会失败。所有测试临时产物位于 `outputs/pytest/`。实验产物位于 `outputs/droid_episode0/`，包括 sample/manifest/audit、analytic commands、geometry/calibration/closed-loop metrics、adapter/PID 参数、完整 traces、report、figures 和 SHA256 清单。

fit `[30,210)`、validation `[220,260)`、test `[270,330)` 保持时序隔离；带历史/延迟的支持不得跨边界。所有标准化只在 fit 计算，模型及增益只在 validation 选择。每个候选、场景、评估区间单独初始化历史、延迟队列和 PID 状态，计分剔除 8 帧 warm-up；部署时跨 chunk 保留 PID 状态。

在线入口是 `s2a.runtime.StateToActionAdapter`：

```python
from s2a.control import PIDConfig
from s2a.runtime import StateToActionAdapter

adapter = StateToActionAdapter(H_FC, dt_command=planned_period, pid_config=PIDConfig())
adapter.set_chunk(measured_tcp, physical_delta_chunk, planned_knot_times)
result = adapter.update(current_measured_tcp, observation_time, now=decision_time)
command = result["command"]  # 6D arm + 绝对 gripper，gripper_action_space="position"
```

示例变量由调用方提供；`H_FC` 来自 tool contract。chunk 必须已反归一化、有有效 mask，时间轴必须是事先生成的执行参考；`now` 与观测时间使用同一物理时钟。`dt_command` 是计划时域，不读取未来观测才知道的 `transition_dt`。默认 PID 增益为零，实验搜索的参数不作为真机部署配置。

canonical TCP 的局部 delta 在 SE(3) 上右乘积分，再完整转换到 flange；DROID `cartesian_velocity` 是归一化接口量，不能当作 m/s、rad/s。夹爪 slot 9 是绝对 opening 命令，与 arm PID 分离。历史控制器版本和 command/state 时刻仍未确认，编码器属于固定公开版本的 reference-interface。

实验矩阵保留解析、线性、P/PI/PD/PID、增益调度、直接逆及残差模型。后三者需要验证同设备的多 episode 数据后进入 P4；初轮未执行独立仿真和真机。后续 LIBERO 仿真不等于 DROID 真机验证。日志回放、局部模型自洽、独立仿真和真机证据分别记录。开发约定见 [AGENTS.md](AGENTS.md)。

同型号跨数据集的解析验证入口：

```bash
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_cross_dataset.py --seed 42
```

名单在 [cross_dataset_franka.json](configs/cross_dataset_franka.json) 冻结，分别评估 RoboMIND-v1 的30条真实Panda日志，以及 M1/A1 的20/10条仿真日志。各自使用原始TCP/方向约定与 source row provenance，不迁移DROID控制器参数。RoboMIND master joint 另经名义Panda FK作命令代理诊断；所有结果仍属于日志验证，未执行真机或独立仿真控制。设计见 [扩展计划](plans/v0.0.1/cross_dataset_franka_plan.md)，最终指标见 `reports/cross_dataset_franka.md`。

LIBERO 的仿真入口使用独立 Python 3.11 环境，固定 MuJoCo2.3.7、robosuite1.4.1、NumPy1.26.4，环境版本见 [snapshot](configs/libero_sim_requirements_20261016.txt)。`src/s2a/osc.py` 针对固定 OSC 内环做二阶响应解析反解，使用当前末端速度并施加保守权重与最终命令变化率限制；不使用 DROID 接口常数。实验方案见 [仿真计划](plans/v0.0.1/libero_simulation_plan.md)。

```bash
# 原生expert正控、相同oracle轨迹与独立小幅运动（CPU物理仿真）
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_libero_matrix.py --seed 42 --stage qualification
# 旧checkpoint已保存预测的固定参考回放，不重新查询模型
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_libero_matrix.py --seed 42 --stage prediction_replay
# 先在空闲GPU上启动5个独立副本（gpu=0..4，端口8180..8184）
LD_LIBRARY_PATH='' .venv/bin/python scripts/serve_libero_policy.py --seed 42 --gpu 0
# 5个副本就绪后执行真实图像闭环，每trial两候选用同一副本
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_libero_matrix.py --seed 42 --stage policy
# 检查所有物理步、控制约束、成对输入与指纹，再生成报告/曲线
LD_LIBRARY_PATH='' .venv/bin/python scripts/summarize_libero_simulation.py --seed 42
```

单trial可直接用 `outputs/libero_env311/bin/python scripts/run_libero_simulation.py --seed 42 --stage oracle --candidate compensated --trial 0`，Python前仍加 `LD_LIBRARY_PATH=''`。新增仿真不读取Cosmos训练dataloader、不修改外部仓库；只有policy服务通过显式路径只读加载Cosmos推理实现和权重。policy仿真不读取专家未来状态。重跑矩阵可复用完全相同配置的已完成项；完整目录已存在但执行中断时需先归档该项，避免覆盖证据。

2026-10-16配置保留历史协议。当前正控使用 `run_libero_audit.py` 显式刷新恢复后的控制器反馈，按case隔离LIBERO环境配置。当前图像闭环使用 `configs/libero_policy_20261019.json`：baseline是原arm+position，候选是保守响应逆+intent，两者都刷新初始化缓存。接口提供可选实验配置，默认夹爪映射仍为position。

```bash
# 65项已见场景的时间/缓存/夹爪/arm因子对照；从clean commit执行
LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/run_libero_audit.py --seed 42 --workers 4

# metadata预先冻结的额外10条：70项普通对照，120项压力，70项完整轨迹/终态
LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/prepare_libero_holdout.py --seed 42
LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/run_libero_audit.py --config configs/libero_holdout_20261019.json --seed 42
LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/run_libero_audit.py --config configs/libero_stress_20261019.json --seed 42
LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/run_libero_audit.py --config configs/libero_terminal_20261019.json --seed 42

# h5py和PyArrow使用各自兼容的解释器，通过outputs中的数字bridge核对R1/R2来源
LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/audit_libero_provenance.py --seed 42 --stage export
LD_LIBRARY_PATH='' .venv/bin/python scripts/audit_libero_provenance.py --seed 42 --stage verify
LD_LIBRARY_PATH='' .venv/bin/python scripts/prepare_libero_published.py --seed 42
LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/run_libero_audit.py --config configs/libero_published_20261019.json --seed 42
LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/run_libero_audit.py --config configs/libero_published_terminal_20261019.json --seed 42

# 保存模型目标的独立前向回放：需先完成本轮实时baseline，和实时图像闭环分开报告
LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/run_libero_audit.py --config configs/libero_policy_replay_20261019.json --seed 42

# 完整性/相同参考前缀/成对模型输入、报告及PNG/PDF
LD_LIBRARY_PATH='' .venv/bin/python scripts/summarize_libero_audit.py --seed 42
LD_LIBRARY_PATH='' .venv/bin/python scripts/plot_libero_audit.py --seed 42
```

每个新矩阵拒绝覆盖已有目录；`--resume`仅复用配置一致、无错误且预算完整的结果。未完成目录先归档再重跑，任务失败的完整结果同样保留。`--allow-incomplete`的报告明确标记未完成，不能作为完整验收通过。

新模型服务沿用 `serve_libero_policy.py`，显式传入 `--config configs/libero_policy_20261019.json --gpu 0 --port 8280 --seed 42`，其余4个副本依次对应GPU1..4/端口8281..8284；执行命令为 `run_libero_matrix.py --config configs/libero_policy_20261019.json --stage policy --seed 42`。检查模型输出合同用 `audit_libero_predictions.py --policy-config configs/libero_policy_20261019.json --seed 42`。同步推理期间暂停物理时钟，不将该评测当作真实延迟下的实时控制证明。

### WAM checkpoint 四层验收（2026-10-19 v2）

本轮以冻结 WAM 权重为被测对象，分别验证训练/推理数值合同、直接 state 预测、S2A执行资格和无推理延迟的多任务闭环。方案、保守决策与完成状态见 [实施记录](plans/v0.0.1/record_20261019v2.md) 和 [结论](plans/v0.0.1/conclusion_20261019v2.md)。

`configs/wam_checkpoint_acceptance_20261019v2.json` 保留原服务协议，`configs/wam_checkpoint_masked_20261019v2.json` 定义显式mask对照：无效Action128维度在初始噪声、网络输入和velocity中始终为零，匹配训练合同。`src/s2a/wam_runtime.py`只在独立模型进程中加载相关依赖，不更改外部Cosmos代码或任何学习参数。6200基线采用实际warm-start父权重及当前run的归一化合同，不能解释为旧TCP协议的原生成绩。

冻结配置中的caption嵌套假设已撤回，正确解释见 [协议勘误](plans/v0.0.1/wam_protocol_errata_20261019v2.md)。保留原文件字节以维持预注册哈希及严格恢复条件。

mask缺陷的源码定位、已执行回归和生产移植边界见 [serving合同说明](reports/wam_serving_contract_20261019v2.md)。当前hook只接受LIBERO十槽合同，其他domain和通用batch仍需在生产移植时单独验证。

所有命令从本仓库执行，产物写入`outputs/wam_checkpoint_acceptance_20261019v2`。已经存在的完整或失败证据均拒绝覆盖；运行前检查目录和服务，重跑使用新输出目录或显式归档。

```bash
# 固定离线episode与多阶段输入；下载并校验新任务的官方数据
LD_LIBRARY_PATH='' .venv/bin/python scripts/prepare_wam_acceptance.py --seed 42
LD_LIBRARY_PATH='' .venv/bin/python scripts/fetch_libero_acceptance_assets.py --seed 42
# 固定6个独立物理初态；执行专家command与真实state正控、独立运动参考
LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/prepare_wam_simulation.py --seed 42
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_wam_simulation_matrix.py --seed 42 --stage native
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_wam_simulation_matrix.py --seed 42 --stage oracle
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_wam_simulation_matrix.py --seed 42 --stage probe
# 对应冻结服务就绪后，每个checkpoint分别运行；一个HTTP服务只接收一个请求
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_wam_state_matrix.py \
  --config configs/wam_checkpoint_masked_20261019v2.json --seed 42 --checkpoint 8800_masked
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_wam_simulation_matrix.py \
  --config configs/wam_checkpoint_masked_20261019v2.json --seed 42 --stage wam --checkpoint 8800_masked
# 无GPU的来源与标签检查，以及明确标注是否完整的汇总
LD_LIBRARY_PATH='' .venv/bin/python scripts/audit_wam_training_history.py --seed 42
LD_LIBRARY_PATH='' .venv/bin/python scripts/audit_wam_state_labels.py --seed 42
LD_LIBRARY_PATH='' .venv/bin/python scripts/analyze_wam_acceptance.py --seed 42 --partial
# 仅在全部预注册矩阵预算完成且完整性校验通过后生成最终汇总
LD_LIBRARY_PATH='' .venv/bin/python scripts/analyze_wam_acceptance.py --seed 42
```

模型服务`serve_wam_acceptance.py`和数值审计`audit_wam_numerics.py`使用外部Cosmos的Python解释器，仅只读加载模型实现；`PYTHONPATH`显式包含外部Cosmos根目录和本仓库`src`，并设置`LD_LIBRARY_PATH=''`、`PYTHONDONTWRITEBYTECODE=1`、独立的`CUDA_VISIBLE_DEVICES`及`MASTER_PORT`。编译缓存通过`TORCHINDUCTOR_CACHE_DIR`放在本仓库`outputs`。服务参数为`--seed 42 --checkpoint 8800 --port 8594 --active-slot-mask --exact-published-caption`；数值审计参数为`--seed 42 --horizon 40 --published-caption`，`--training-backend`额外复算保存的attention和固定padding配置，需要独占GPU显存。`audit_wam_caption_contract.py`在同一外部解释器中只做CPU输入准备核验。

在线只输入当前图像和简短任务指令，physics在推理期间暂停；成功以LIBERO原生谓词判定，分别报告曾成功、末端成功和抓取接触/高度/位移等连续进度。完整caption属于离线特权诊断。Study-heldout仅表示本次实验整episode隔离，当前没有证据可将其称为预训练未见数据。

完整性检查与原生窗口补充：

```bash
# 完整权重哈希只执行一次；DCP存储核验使用具备相同torch版本的外部模型解释器
LD_LIBRARY_PATH='' .venv/bin/python scripts/verify_wam_acceptance.py --seed 42 --stage fingerprint
# 主矩阵全量完成、生成summary.json之后核对每条已保存响应和执行目标
LD_LIBRARY_PATH='' .venv/bin/python scripts/verify_wam_acceptance.py --seed 42 --stage verify
# 以下prepare仅用于全新补充输出目录，复用主实验的固定fixture和权重指纹
LD_LIBRARY_PATH='' .venv/bin/python scripts/summarize_wam_native_horizon.py --seed 42 --prepare
# 独占同权重masked HTTP服务后，40步生成、仍执行前8步，完整12个用例
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_wam_simulation_matrix.py \
  --config configs/wam_checkpoint_native_h40_20261019v2.json --seed 42 --stage wam --checkpoint 8800_masked
LD_LIBRARY_PATH='' .venv/bin/python scripts/summarize_wam_native_horizon.py --seed 42
LD_LIBRARY_PATH='' .venv/bin/python scripts/verify_wam_acceptance.py \
  --config configs/wam_checkpoint_native_h40_20261019v2.json --seed 42 --stage verify
LD_LIBRARY_PATH='' .venv/bin/python scripts/plot_wam_acceptance.py --seed 42
LD_LIBRARY_PATH='' .venv/bin/python scripts/report_wam_checkpoint_acceptance.py --seed 42
```

H16是四checkpoint的同协议主对照，H40是最新8800的训练原生窗口诊断；二者分开报告。`--job-indices`只分割不重叠的用例集合，`--server-port`可接入已验证逐项复现相同输出的服务副本，不改变采样和控制参数。`scripts/schedule_wam_replica.py`是本轮已记录进程与分组的专用资源调度入口，不作为新机器上的通用启动器。

最终报告入口要求主矩阵和H40补充的完整汇总及独立完整性检查均完成，重新核对已保存文件指纹和90条/51480物理步预算，拒绝覆盖报告。任务谓词、参考跟踪及目标物体接触/高度/位移分别呈现，不能将位移或接触自动计作成功。
