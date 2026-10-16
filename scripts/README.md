# 实验入口

后续抽取、解析转换、辨识、整定和评估脚本放在此目录，从仓库根目录执行。当前还没有实验 CLI；入口实现后在根 README 中补齐实际运行命令。

算法放在 `src/s2a/`，脚本负责读取 `configs/`、组织运行及输出报告。数据源只读，相对输出路径以仓库根目录解析，默认写 `outputs/<experiment_name>/`。

每次运行保存 seed、配置、代码版本、数据指纹和指标，区分历史回放、局部模型 rollout、独立仿真及真机执行。计划见 [001_state_to_action.md](../plans/v0.0.1/001_state_to_action.md)。
