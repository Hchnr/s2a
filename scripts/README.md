# 实验入口

直接比较WAM参考与真实机器人state：`LD_LIBRARY_PATH='' .venv/bin/python scripts/audit_libero_wam_states.py --seed 42`，配置为`configs/libero_wam_state_comparison_20261019.json`。复用保存的10份同源图像预测，只读R2的observation.state，不读取专家action；输出位置/SO(3)误差、开度误差、保持初态基线及指纹。不会重新推理或跑物理；已有输出拒绝覆盖，复跑须另设output_dir。

从仓库根执行 `LD_LIBRARY_PATH='' .venv/bin/python scripts/run_experiment.py --seed 42 --stage all --plots`。

`--config` 指定 JSON；`--seed` 必须显式提供并与配置相同；stage 可选 `extract`、`analytic`、`calibrate`、`all`（默认）或 `report`。前三者分别运行到 P0、P1、P2；`all` 包含 P3 与 Markdown/JSON 报告；`report` 只使用已有指标。`--plots` 需要 `.[report]`，生成曲线不进行新的参数选择。

算法位于 `src/s2a/`，入口负责组织运行。原始数据只读，不访问视频，不启动 Cosmos dataloader。相对 output_dir 以 S2A 根目录解析并限制在 `outputs/` 内。重跑会覆盖同名实验产物；需保留另一轮时使用新的 output_dir。

样本 provenance、SHA256、代码版本、依赖版本、网格和 seed 随产物保存。实验合同及证据分级见 [计划](../plans/v0.0.1/001_state_to_action.md) 与 [记录](../plans/v0.0.1/record_20261016.md)。

`LD_LIBRARY_PATH='' .venv/bin/python scripts/verify_artifacts.py --seed 42` 检查保存的产物/源码指纹、test rollout 完整性及所有最终送出命令的范数/变化率界。完整性通过与控制性能验收分别报告。

`run_cross_dataset.py --seed 42` 执行冻结的60条Panda跨数据集轨迹，使用各自TCP合同验证解析式，并单列原始Cartesian action或nominal master-joint FK差异。结果写 `outputs/cross_dataset_franka/`；只读原始数字列，不执行控制。RoboMIND-v1为真实日志，M1/A1为仿真日志。各自分类、指纹与未验证项随报告保存。

`run_libero_simulation.py` 是独立MuJoCo单trial入口；`run_libero_matrix.py` 组织 `qualification`、`prediction_replay` 或 `policy` 成对矩阵。仿真子进程固定用S2A的 `outputs/libero_env311`，每步实际调用环境前向控制，并保存当前状态、已生成目标、执行命令及执行后状态。`serve_libero_policy.py` 等待空闲GPU和其他训练结束，从S2A工作目录只读加载固定Cosmos权重；5个副本分别指定 `--gpu 0` 至 `--gpu 4`，端口和rendezvous互相独立。启动器是前台监控进程，服务PID保存在outputs中，仅清理本轮拥有的进程。

`summarize_libero_simulation.py --seed 42` 独立核查37项实际仿真的物理时钟、state error、最终命令幅值/变化率、固定参考成对一致性和源码/输入/产物指纹。`--allow-incomplete` 只生成明确标记未完成的阶段报告，不将缺失项计为通过。完整性校验与任务成功率分开判断。所有命令前使用 `LD_LIBRARY_PATH=''`。

2026-10-19入口：`run_libero_audit.py`按显式config执行case×episode物理矩阵，支持严格`--resume`；`prepare_libero_holdout.py`和`prepare_libero_published.py`生成事先冻结的fixture。`audit_libero_provenance.py`分别在h5py解释器export与PyArrow解释器verify，按整段命令内容核对来源，再分别评价HDF5/R1/R2观测。`audit_libero_predictions.py`独立核对所有已完成模型响应的单位/归一化/mask/时钟；`summarize_libero_audit.py`核对当前矩阵、执行状态、真实前向误差和代码/输入/输出指纹；`plot_libero_audit.py`生成可导出的PNG/PDF。具体命令和证据边界见根README与 `plans/v0.0.1/record_20261019.md`。


WAM 效果专项使用 `configs/libero_wam_effect_20261019.json`，五个服务分别在 GPU 0–4/8480–8484；分量干预使用 `configs/libero_wam_components_20261019.json`，GPU 5–7/8485–8487。先按各配置的 `server_urls` 启动 `serve_libero_policy.py --config <config> --seed 42 --gpu <gpu> --port <port>` 并检查 `/info` 的权重路径。权重、推理代码和数据保持外部只读。依赖先前来源审计的 published bridge；复跑需生成该 bridge 或通过配置提供已校验的同源参考，并修改 output_dir，所有已有物理证据拒绝覆盖。

```bash
LD_LIBRARY_PATH='' .venv/bin/python scripts/fingerprint_libero_checkpoint.py --config configs/libero_wam_effect_20261019.json --seed 42
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_libero_effect.py --seed 42 --stage controls
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_libero_effect.py --seed 42 --stage wam
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_libero_effect.py --config configs/libero_wam_components_20261019.json --seed 42 --stage components --workers 3
LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/prepare_libero_contact_bounds.py --seed 42
LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/prepare_libero_workspace_bound.py --seed 42
# 待 oracle_response 与 wam_affine 的完整原始轨迹生成后运行；变慢的冻结参考重放只作诊断
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_libero_endpoint_diagnostic.py --seed 42
LD_LIBRARY_PATH='' .venv/bin/python scripts/summarize_libero_effect.py --seed 42
LD_LIBRARY_PATH='' .venv/bin/python scripts/plot_libero_effect.py --seed 42
LD_LIBRARY_PATH='' .venv/bin/python scripts/export_libero_effect_video.py --seed 42
```

同步入口验证 HTTP 前后完整物理状态和仿真时间不变，全部前 8 个目标真实执行。分量干预在 world/source 点替换分量、保留 WAM 原始输出；含 oracle 的成功不计为纯 WAM 成功。独立汇总按冻结名单核对完整预算、逐目标执行、端点连续性、物理位姿、命令边界/变化率、模型反归一化与图像指纹；`--stage controls` 等仅保存阶段核验，不生成全量完成报告。

效果专项最终55项/50739步与2010个响应核验通过，完整结果见 [报告](../reports/libero_wam_effect_20261019.md)。准静态 `--resume` 仅复用同配置/输入SHA不变的完整case；新运行已按300×12覆盖环境horizon。视频导出需可选 `imageio-ffmpeg`，当前环境已提供；输出来自真实输入相机，拒绝覆盖已有视频。核心源码应先提交，独立核验会按运行记录commit核对实际执行SHA。

当前GPU限制以根AGENTS.md为准：只允许物理4–7，以上GPU0–3命令为历史记录。已完成初态的成对轨迹诊断可运行 `CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/plot_wam_scene_diagnostics.py --record task0_init0 --seed 42`。入口先独立核验该初态所有checkpoint和seed的完整预算，输出TCP/SO(3)、夹爪、原生抓取接触和任务判定曲线及PNG/PDF和原始证据SHA，已有输出拒绝覆盖。前5秒几何图与分段误差是事后诊断窗口，整条30秒曲线和成功判定一并保留；场景图不代表全矩阵验收或因果证明。

`audit_wam_gripper_commands.py --seed 42 --output outputs/wam_independent_20261020/gripper_command_audit_<snapshot>.json`只读复核增量完整性SHA，比较实际夹爪命令/方向反转、接触和控制限制，区分command作用起点与endpoint；未完成项不计失败。`run_wam_gripper_replay.py --seed 42 --stage prepare/baseline/counterfactual/report`使用configs/wam_gripper_replay_20261020.json做三源×三夹爪的CPU固定参考重放；先通过原intent复现门槛再解释其他策略。命令均加`CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/`前缀，重放是事后执行反事实，不计作新WAM闭环成功。
