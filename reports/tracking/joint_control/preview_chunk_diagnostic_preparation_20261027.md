# 参考窗口诊断的输入冻结与夹爪历史重建

`scripts/prepare_preview_chunk_diagnostic.py` 已在原生Python3.11实际退出0，新增模型/物理/fresh均0。准备证据 `outputs/preview_chunk_diagnostic_preparation_20261027/prepared.json`，SHA `79d07e233349af103ba935f9d968cc1de0ef7049e7aa2bbdfe9bc9869da30d80`。

按预登记条件选择每个完整案例的最早匹配查询：当前开度>0.8、完整40步参考包含开度<0.2、首次闭合索引>=8；7个匹配、wine/raw/seed43没有匹配，明确记录不运行该诊断，未补选其他时刻。

| 案例 | 查询索引 | 前缀物理步 | 首次闭合参考索引 |
|---|---:|---:|---:|
| task0_init0_s42_raw | 11 | 88 | 39 |
| task0_init0_s42_terminal058 | 3 | 24 | 37 |
| task0_init0_s43_raw | 3 | 24 | 30 |
| task0_init0_s43_terminal058 | 1 | 8 | 35 |
| task1_init0_s42_raw | 8 | 64 | 30 |
| task1_init0_s42_terminal058 | 16 | 128 | 34 |
| task1_init0_s43_terminal058 | 13 | 104 | 29 |

已通过完整原夹爪模型和latch重建选取点前共440步的历史，逐步核对命令误差≤1e-12及route一致。未将夹爪缓存清零或从开度猜测内部状态。这里是日志重算证据，尚未完成物理分支状态恢复。

最大重建命令差为 0。两分支各自重放前缀，之后分别执行8/40步，总物理预算为1216步。绑定原worker、chunk、step、config、模型参数、manifest、工具坐标合同和端点receipt。

物理分支执行入口和共同前8步核对尚待完成，prepared_only=true、execution_authorized=false。使用原冻结求解器进行该诊断，出现数值异常就保留失败，不热替换新候选。新零空间求解器仍需独立控制资格。整个诊断不授予600步任务成功或fresh资格。
