# 实验入口

直接比较WAM参考与真实机器人state：`LD_LIBRARY_PATH='' .venv/bin/python scripts/audit_libero_wam_states.py --seed 42`，配置为`configs/libero_wam_state_comparison_20261019.json`。复用保存的10份同源图像预测，只读R2的observation.state，不读取专家action；输出位置/SO(3)误差、开度误差、保持初态基线及指纹。不会重新推理或跑物理；已有输出拒绝覆盖，复跑须另设output_dir。

从仓库根执行 `LD_LIBRARY_PATH='' .venv/bin/python scripts/run_experiment.py --seed 42 --stage all --plots`。

`--config` 指定 JSON；`--seed` 必须显式提供并与配置相同；stage 可选 `extract`、`analytic`、`calibrate`、`all`（默认）或 `report`。前三者分别运行到 P0、P1、P2；`all` 包含 P3 与 Markdown/JSON 报告；`report` 只使用已有指标。`--plots` 需要 `.[report]`，生成曲线不进行新的参数选择。

算法位于 `src/s2a/`，入口负责组织运行。原始数据只读，不访问视频，不启动 Cosmos dataloader。相对 output_dir 以 S2A 根目录解析并限制在 `outputs/` 内。重跑会覆盖同名实验产物；需保留另一轮时使用新的 output_dir。

样本 provenance、SHA256、代码版本、依赖版本、网格和 seed 随产物保存。实验合同及证据分级见 [计划](../plans/v0.0.1/001_state_to_action.md) 与 [记录](../plans/v0.0.1/record_20261016.md)。

`LD_LIBRARY_PATH='' .venv/bin/python scripts/verify_artifacts.py --seed 42` 检查保存的产物/源码指纹、test rollout 完整性及所有最终送出命令的范数/变化率界。完整性通过与控制性能验收分别报告。

`run_cross_dataset.py --seed 42` 执行冻结的60条Panda跨数据集轨迹，使用各自TCP合同验证解析式，并单列原始Cartesian action或nominal master-joint FK差异。结果写 `outputs/cross_dataset_franka/`；只读原始数字列，不执行控制。RoboMIND-v1为真实日志，M1/A1为仿真日志。各自分类、指纹与未验证项随报告保存。

`run_libero_simulation.py` 是独立MuJoCo单trial入口；`run_libero_matrix.py` 组织 `qualification`、`prediction_replay` 或 `policy` 成对矩阵。仿真子进程固定用S2A的 `outputs/libero_env311`，每步实际调用环境前向控制，并保存当前状态、已生成目标、执行命令及执行后状态。`serve_libero_policy.py` 等待空闲GPU和其他训练结束，从S2A工作目录只读加载固定Cosmos权重；后续显式指定物理 `--gpu 4`、`5`、`6` 或 `7`，最多四个并行副本，端口和rendezvous互相独立。历史五副本映射不可直接复用。启动器是前台监控进程，服务PID保存在outputs中，仅清理本轮拥有的进程。

`summarize_libero_simulation.py --seed 42` 独立核查37项实际仿真的物理时钟、state error、最终命令幅值/变化率、固定参考成对一致性和源码/输入/产物指纹。`--allow-incomplete` 只生成明确标记未完成的阶段报告，不将缺失项计为通过。完整性校验与任务成功率分开判断。所有命令前使用 `LD_LIBRARY_PATH=''`。

2026-10-19入口：`run_libero_audit.py`按显式config执行case×episode物理矩阵，支持严格`--resume`；`prepare_libero_holdout.py`和`prepare_libero_published.py`生成事先冻结的fixture。`audit_libero_provenance.py`分别在h5py解释器export与PyArrow解释器verify，按整段命令内容核对来源，再分别评价HDF5/R1/R2观测。`audit_libero_predictions.py`独立核对所有已完成模型响应的单位/归一化/mask/时钟；`summarize_libero_audit.py`核对当前矩阵、执行状态、真实前向误差和代码/输入/输出指纹；`plot_libero_audit.py`生成可导出的PNG/PDF。具体命令和证据边界见根README与 `plans/v0.0.1/record_20261019.md`。


WAM 效果专项使用 `configs/libero_wam_effect_20261019.json`，历史五服务映射为 GPU 0–4/8480–8484；分量干预使用 `configs/libero_wam_components_20261019.json`，历史映射为 GPU 5–7/8485–8487。这些是已完成实验的资源记录，不能直接重启旧八卡编排器。复跑必须先明确重排至物理GPU4–7并记录新映射；服务命令为 `CUDA_VISIBLE_DEVICES=<允许卡> LD_LIBRARY_PATH='' .venv/bin/python scripts/serve_libero_policy.py --config <config> --seed 42 --gpu <允许卡> --port <port>`，随后检查 `/info` 的权重路径。权重、推理代码和数据保持外部只读。依赖先前来源审计的 published bridge；复跑需生成该 bridge 或通过配置提供已校验的同源参考，并修改 output_dir，所有已有物理证据拒绝覆盖。

```bash
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/fingerprint_libero_checkpoint.py --config configs/libero_wam_effect_20261019.json --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/run_libero_effect.py --seed 42 --stage controls
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/run_libero_effect.py --seed 42 --stage wam
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/run_libero_effect.py --config configs/libero_wam_components_20261019.json --seed 42 --stage components --workers 3
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/prepare_libero_contact_bounds.py --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/prepare_libero_workspace_bound.py --seed 42
# 待 oracle_response 与 wam_affine 的完整原始轨迹生成后运行；变慢的冻结参考重放只作诊断
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/run_libero_endpoint_diagnostic.py --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/summarize_libero_effect.py --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/plot_libero_effect.py --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/export_libero_effect_video.py --seed 42
```

同步入口验证 HTTP 前后完整物理状态和仿真时间不变，全部前 8 个目标真实执行。分量干预在 world/source 点替换分量、保留 WAM 原始输出；含 oracle 的成功不计为纯 WAM 成功。独立汇总按冻结名单核对完整预算、逐目标执行、端点连续性、物理位姿、命令边界/变化率、模型反归一化与图像指纹；`--stage controls` 等仅保存阶段核验，不生成全量完成报告。

效果专项最终55项/50739步与2010个响应核验通过，完整结果见 [报告](../reports/libero_wam_effect_20261019.md)。准静态 `--resume` 仅复用同配置/输入SHA不变的完整case；新运行已按300×12覆盖环境horizon。视频导出需可选 `imageio-ffmpeg`，当前环境已提供；输出来自真实输入相机，拒绝覆盖已有视频。核心源码应先提交，独立核验会按运行记录commit核对实际执行SHA。

当前GPU限制以根AGENTS.md为准：只允许物理4–7，以上GPU0–3命令为历史记录。已完成初态的成对轨迹诊断可运行 `CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/plot_wam_scene_diagnostics.py --record task0_init0 --seed 42`。入口先独立核验该初态所有checkpoint和seed的完整预算，输出TCP/SO(3)、夹爪、原生抓取接触和任务判定曲线及PNG/PDF和原始证据SHA，已有输出拒绝覆盖。前5秒几何图与分段误差是事后诊断窗口，整条30秒曲线和成功判定一并保留；场景图不代表全矩阵验收或因果证明。

`audit_wam_gripper_commands.py --seed 42 --output outputs/wam_independent_20261020/gripper_command_audit_<snapshot>.json`只读复核增量完整性SHA，比较实际夹爪命令/方向反转、接触和控制限制，区分command作用起点与endpoint；未完成项不计失败。`run_wam_gripper_replay.py --seed 42 --stage prepare/baseline/counterfactual/report`使用configs/wam_gripper_replay_20261020.json做三源×三夹爪的CPU固定参考重放；先通过原intent复现门槛再解释其他策略。命令均加`CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/`前缀，重放是事后执行反事实，不计作新WAM闭环成功。

`report_wam_independent.py --seed 42`要求主144条、执行16/40各36条的完整summary/integrity，以及9条CPU重放完整证据；任一缺失拒绝生成正式报告。报告按目标原生双侧指垫接触及任务判定划分失败阶段，单列固定参考反事实并复核来源与结果SHA，不将阶段或跟踪统计写成唯一因果。

`finalize_wam_independent.py --seed 42 --check-only`只读检查上述216条结果及当前分区/调度器退出门槛，pending不算模型失败。去掉`--check-only`会独占新建outputs/wam_independent_20261020/finalizer/，每30秒检测，全部就绪后CPU串行汇总和生成报告；stdout应重定向到该目录外。命令使用上述CUDA空及LD_LIBRARY_PATH空前缀；已有正式产物或旧finalizer目录、错误、取消、冻结输入变化均拒绝，需审阅保留证据后另行处理，不自动重试、覆盖或Git提交。

wine补充执行诊断使用`run_wam_gripper_replay.py --config configs/wam_wine_gripper_replay_20261020.json --seed 42 --stage prepare/baseline/counterfactual/report`，仍逐阶段先复现原intent；独立输出和报告保护旧9条。重放报告路径可显式配置且必须在本仓库reports内。此2源×3夹爪CPU实验只重放原执行前缀，不执行原H40后缀、不重新推理，不并入独立WAM成功率，也不替代原定执行窗口对照。

2026-10-21稳定性阶段入口如下。所有命令从本仓库执行，CPU入口加`CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/`前缀。模型由自有编排设置单张物理GPU4–7；新运行须复制配置并使用新的output_dir，已有证据不覆盖。

| 入口 | 用途与边界 |
| --- | --- |
| `run_wam_stability_screen.py --config <config> --seed 42 --fingerprints <complete-byte-manifest>` | 单服务48个固定diagnostic请求、身份/mask/权重检查、GPU锁与自有进程清理；`--dry-run`不加载模型。 |
| `run_wam_stability_service.py --config <config> --seed 42 --checkpoint <name> --variants <comma-separated> --fingerprints <manifest> --matrix-entry scripts/run_wam_recovery_matrix.py` | 排队运行冻结恢复执行器；每次推理暂停物理，保存raw/effective/admitted参考和真实状态/命令；不是实时性能评测。 |
| `audit_wam_stability_pilots.py --seed 42 --output <new-snapshot.json>` | 独立重建首24条的控制/夹爪/参考时钟、完整预算与历史逐值baseline；pending不能计失败。 |
| `audit_wam_recovery.py --output <new-snapshot.json>` | 固定6oracle+24WAM恢复矩阵，独立重建恢复时钟、输出正则、motor目标、不同参考时钟误差及任务结果；`--reuse-verified <snapshot>`只复用同审计源码且全部输入SHA仍相同的case。 |
| `summarize_wam_earlier_screen.py --output <new-snapshot.json>` | 8400/8600/8800/15400同guidance1.0原始响应与state计分；四组全部48请求完成才给比较，`audit_wam_earlier_screen.py`另用直接积分独立复核。 |
| `analyze_wam_seed_consensus.py --output <new-snapshot.json>` | 只读缓存两seed等权world参考中点，0新增推理/物理；`audit_wam_seed_consensus.py --output <new-audit.json>`用Slerp独立复核，不把state误差当任务成功。 |
| `qualify_gripper_position_hold.py --config configs/gripper_position_hold_qualification_20261021.json --seed 42 --prepare-only` | 冻结004九条CPU资格及实际依赖；去掉prepare-only执行，先精确复现旧v2，再评价新v3两方式；已完成目录拒绝重跑。 |
| `run_wam_stability_service.py ... --matrix-entry scripts/run_wam_position_matrix.py` | 004资格门禁通过后的005/006新夹爪执行入口，物理函数与oracle共用，模型单卡、每服务一CPU worker；须沿用各自预注册预算。 |
| `audit_wam_position.py --output <new-snapshot.json>` | 固定9oracle＋24WAM的005独立重建，新增独立scalar v3、qpos/motor/开度连续及对应003首query精确回归；不包含006。 |
| `audit_wam_earlier_position.py --output <new-snapshot.json>` | 006固定8400十二条独立重建；仅引用已通过的9oracle资格，首query对同8400旧十二条，不把资格重复计数或扩大005分母。 |
| `analyze_wam_position_gripper.py --audit <audit.json> --output <new-snapshot.json>` | 对已验证005与旧003仅共同首query作严格开度配对；600步对照另标后续输入分叉，不能当全程相同参考。 |
| `report_wam_position.py --audit <complete-audit.json> --gripper-diagnostic <complete-diagnostic.json> --output <new-directory>` | 仅005全部33项verified和24对诊断完整时生成正式报告/图；partial门禁不通过，不提前给全矩阵结论。 |

`diagnose_wam_reference_geometry.py --seed 42`与`review_wam_reference_geometry.py`只读固定6条raw intent_hold、其旧baseline及两条恢复bowl，分别记录参考几何、实际提前闭合、model时间消费和对象指标；图像只覆盖已保存观测，不能推断连续接触机制。`qualify_osc_rotation.py`、`qualify_gripper_tracking.py`、`qualify_gripper_intent_hold.py`及`qualify_reference_recovery.py`属于独立物理资格，oracle/合成参考均不计入WAM成功率。完整协议、预算、资格失败和未执行项见[002计划](../plans/v0.0.1/002_wam_stability_20261021.md)、[003恢复对照](../plans/v0.0.1/003_wam_recovery_20261021.md)与[本轮记录](../plans/v0.0.1/record_20261021.md)。


010独立短重观测：`CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/run_wam_short_replan.py --interval 8 --seed 42`（物理GPU4/9444）或`--interval 16`（GPU5/9445）。每服务顺序运行alpha1/alpha0的两配置，单CPU物理worker；全部12条/7200步/最多678请求。`--dry-run`复核先决与配置/完整权重stat，0模型/物理；固定output已运行时拒绝覆盖。新入口只编排冻结的position执行器，alpha0改变模型参考、不是原始checkpoint成绩。启动前需完整005/006/007/009证据；完整协议见[010](../plans/v0.0.1/010_short_replan_wam_20261021.md)。

011/013 state专项使用隔离的`run_wam_sampler_steps.py`与`run_wam_backend_screen.py`；先前者48份真实35-step全值exact，再在首次velocity前配对actual sampler state/conditions。`audit_wam_sampler_steps.py`为冻结严格入口，`audit_wam_sampler_steps_release.py`为显式窄退出旁证版；`audit_wam_backend_state.py --shutdown-supplement`同样明确内部finally收据不可得，原strict拒绝保留，不能冒称复原ledger。各入口的预算/先决见011/013计划；不自动扩大到物理或fresh。

014的`wam_position_acceptance_checker.py`是seed与首query来源回调显式参数化的字面旧checker，`verify_wam_acceptance_seed_regression.py`只读回归旧27条。015的`audit_wam_seed43_qualification.py`独立验收三旧场景真实43 oracle；这些都不启动WAM或证明43模型任务能力。

016新版本`run_wam_tracking_matrix.py --config configs/tracking_replan_qualification_20261021_v2.json --seed 42 --source oracle --variant position_hold --workers 2 --dry-run`会实际只读验证三条native参考，0环境步进/0模型。去掉dry-run的资格已有输出后拒绝覆盖。新`run_wam_tracking_service.py --seed 42 --gate <frozen-launch-gate.json> --gate-sha256 <sha>`只在完整独立资格与013正常释放等输入SHA门禁通过后启动物理GPU7/9597的一份8400和单物理worker，3条/1800步/max225请求；`--dry-run`也严格核门禁。所有CPU命令沿用空CUDA/LD前缀，原position/recovery入口字节保留；完整预算/算法见016计划。
