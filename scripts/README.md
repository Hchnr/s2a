# 实验入口

从仓库根执行 `LD_LIBRARY_PATH='' .venv/bin/python scripts/run_experiment.py --seed 42 --stage all --plots`。

`--config` 指定 JSON；`--seed` 必须显式提供并与配置相同；stage 可选 `extract`、`analytic`、`calibrate`、`all`（默认）或 `report`。前三者分别运行到 P0、P1、P2；`all` 包含 P3 与 Markdown/JSON 报告；`report` 只使用已有指标。`--plots` 需要 `.[report]`，生成曲线不进行新的参数选择。

算法位于 `src/s2a/`，入口负责组织运行。原始数据只读，不访问视频，不启动 Cosmos dataloader。相对 output_dir 以 S2A 根目录解析并限制在 `outputs/` 内。重跑会覆盖同名实验产物；需保留另一轮时使用新的 output_dir。

样本 provenance、SHA256、代码版本、依赖版本、网格和 seed 随产物保存。实验合同及证据分级见 [计划](../plans/v0.0.1/001_state_to_action.md) 与 [记录](../plans/v0.0.1/record_20261016.md)。

`LD_LIBRARY_PATH='' .venv/bin/python scripts/verify_artifacts.py --seed 42` 检查保存的产物/源码指纹、test rollout 完整性及所有最终送出命令的范数/变化率界。完整性通过与控制性能验收分别报告。

`run_cross_dataset.py --seed 42` 执行冻结的60条Panda跨数据集轨迹，使用各自TCP合同验证解析式，并单列原始Cartesian action或nominal master-joint FK差异。结果写 `outputs/cross_dataset_franka/`；只读原始数字列，不执行控制。RoboMIND-v1为真实日志，M1/A1为仿真日志。各自分类、指纹与未验证项随报告保存。

`run_libero_simulation.py` 是独立MuJoCo单trial入口；`run_libero_matrix.py` 组织 `qualification`、`prediction_replay` 或 `policy` 成对矩阵。仿真子进程固定用S2A的 `outputs/libero_env311`，每步实际调用环境前向控制，并保存当前状态、已生成目标、执行命令及执行后状态。`serve_libero_policy.py` 等待空闲GPU和其他训练结束，从S2A工作目录只读加载固定Cosmos权重；5个副本分别指定 `--gpu 0` 至 `--gpu 4`，端口和rendezvous互相独立。启动器是前台监控进程，服务PID保存在outputs中，仅清理本轮拥有的进程。

`summarize_libero_simulation.py --seed 42` 独立核查37项实际仿真的物理时钟、state error、最终命令幅值/变化率、固定参考成对一致性和源码/输入/产物指纹。`--allow-incomplete` 只生成明确标记未完成的阶段报告，不将缺失项计为通过。完整性校验与任务成功率分开判断。所有命令前使用 `LD_LIBRARY_PATH=''`。

2026-10-19入口：`run_libero_audit.py`按显式config执行case×episode物理矩阵，支持严格`--resume`；`prepare_libero_holdout.py`和`prepare_libero_published.py`生成事先冻结的fixture。`audit_libero_provenance.py`分别在h5py解释器export与PyArrow解释器verify，按整段命令内容核对来源，再分别评价HDF5/R1/R2观测。`audit_libero_predictions.py`独立核对所有已完成模型响应的单位/归一化/mask/时钟；`summarize_libero_audit.py`核对当前矩阵、执行状态、真实前向误差和代码/输入/输出指纹；`plot_libero_audit.py`生成可导出的PNG/PDF。具体命令和证据边界见根README与 `plans/v0.0.1/record_20261019.md`。
