# seed 43 当前图像第二窗口的输入准入

2026-11-13。[预登记](../../../plans/v0.0.1/tracking/references/reset_seed43_live_second_window_20261113.md)
的只读准入已加载 `task0_init0_s43_raw` 的历史复位、原参考和分段首窗，绑定
先前独立重放及真实 seed 43 参考等价审计。历史首窗第 8 端点的状态 SHA 为
`622c20969db1ea57c878716774c062cb7a961eac1b035560e9fa12d710ee3e61`；实际运行
仍必须在本次 8 周期完成后重新测得相同状态，再使用本次当前图像。

准入固定单一任务、seed 43、模型指纹和 1 请求/35 去噪预算；检查模型源码
清单 1004 项、权重清单 163 项及历史输入引用。实际执行前仍须逐文件重算哈希、
完成当前进程和 GPU 准入。历史试验源码保持不变。

32 并发相关回归 **58 项通过、0 skip、0 collection errors**；源码前后一致，
全部测试 worker 实际 wait/reap。结果
`outputs/qa/reset_seed43_live_admission_20261113/result.json` SHA-256
`d0c9ae3e6d93de979896e50ca93f903bf15535f0d55118b4bdc3926b10dd3b37`。
此阶段没有模型初始化、WAM 请求、物理步或最终留出运行；现场接线、独立重放
和真实试验尚未执行。
