# 入口与独立审核

| 领域 | 当前实际入口与审核模块数（不含 __init__） |
| --- | ---: |
| [data](data/index.md) | 26 |
| [maintenance](maintenance/index.md) | 2 |
| [simulation](simulation/index.md) | 82 |
| [tracking](tracking/index.md) | 467 |
| [verification](verification/index.md) | 626 |
| [wam](wam/index.md) | 286 |

从仓库根以 `python -m scripts.<领域>.<入口>` 执行；配置、seed、预算、输出目录和来源必须显式登记。独立审核与生产实现分别保留，不用生产求解器证明其自身正确。

当前 full-H12 输入准入入口为 `scripts.tracking.trajectory_search.probe_tracking_v5_state_vector_dual_recovery --dry` 和 `scripts.tracking.trajectory_search.check_tracking_v5_state_vector_dual_recovery --dry`。CPU 命令显式设置空 `CUDA_VISIBLE_DEVICES`、空 `LD_LIBRARY_PATH`；完整单线程环境与解释器要求见[迁移记录](../docs/migrations/README.md)。`--dry`、导入成功和历史记录均不授权启动 GPU、恢复旧实验或授予控制资格。

历史具体命令见[原始 scripts README](../docs/history/scripts_README_20261029.md)；其中旧 GPU 映射和旧路径是历史声明，不能直接重启。
