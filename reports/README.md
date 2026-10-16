# 实验报告

这里保存 Git 内的精简报告、指标摘要和完成状态，完整产物位于 `outputs/`。

本轮报告为 `droid_episode0_20261016.md`，摘要为 `droid_episode0_20261016.json`；对应 [实施记录](../plans/v0.0.1/record_20261016.md) 和 [结论](../plans/v0.0.1/conclusion_20261016.md)。所有实验均标明代码/数据指纹、控制接口、划分与证据范围。

重建报告：`LD_LIBRARY_PATH='' .venv/bin/python scripts/run_experiment.py --seed 42 --stage report --plots`。它读取已有指标，不重新选择模型或增益。图在 `outputs/droid_episode0/figures/`，报告链接保留本地完整产物的位置。

首轮仅有日志、几何和单 episode 局部模型内证据；增益调度/小网络、独立仿真与真机未执行。不能以日志 action 误差或局部模型的自洽宣称控制成功。
