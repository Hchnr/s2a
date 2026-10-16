# 实验入口

从仓库根执行 `LD_LIBRARY_PATH='' .venv/bin/python scripts/run_experiment.py --seed 42 --stage all --plots`。

`--config` 指定 JSON；`--seed` 必须显式提供并与配置相同；stage 可选 `extract`、`analytic`、`calibrate`、`all`（默认）或 `report`。前三者分别运行到 P0、P1、P2；`all` 包含 P3 与 Markdown/JSON 报告；`report` 只使用已有指标。`--plots` 需要 `.[report]`，生成曲线不进行新的参数选择。

算法位于 `src/s2a/`，入口负责组织运行。原始数据只读，不访问视频，不启动 Cosmos dataloader。相对 output_dir 以 S2A 根目录解析并限制在 `outputs/` 内。重跑会覆盖同名实验产物；需保留另一轮时使用新的 output_dir。

样本 provenance、SHA256、代码版本、依赖版本、网格和 seed 随产物保存。实验合同及证据分级见 [计划](../plans/v0.0.1/001_state_to_action.md) 与 [记录](../plans/v0.0.1/record_20261016.md)。

`LD_LIBRARY_PATH='' .venv/bin/python scripts/verify_artifacts.py --seed 42` 检查保存的产物/源码指纹、test rollout 完整性及所有最终送出命令的范数/变化率界。完整性通过与控制性能验收分别报告。
