# 六案例自适应开发矩阵准备

2026-11-13。按照[固定协议](../../../plans/v0.0.1/tracking/references/reset_multitask_adaptive_600_20261113.md)，
复用相同复位反馈、seed-aware WAM 服务及自适应重规划，接入三任务各两个 seed、
每条最多 600 周期的统一开发入口。每案例的现场模型与仿真由独立进程监督，完成后
可从 reset 独立重放；四个不同 GPU UUID 的首批案例关闭并释放资源后再运行第二批。
统一预检对六案例只哈希一次共享模型和权重，每个案例仍绑定自己的初态、seed、
输入、源码、GPU 和独占输出；异常、早停和未启动案例保留原分母。

32 并发专项中，真实六案例准入及共享流接线 50 项、完整调度及交接 43 项、共享
预检及案例准入 25 项分别通过，均无跳过或收集错误。共享预检首次测试有 1 项失败、
24 项通过：测试替身未将 QA 路径指向临时测试收据；更正后 25 项通过。原失败保留：

| 记录 | SHA-256 |
| --- | --- |
| `outputs/qa/reset_multitask_adaptive_admission_20261113/result.json` | `c086d903bf8f3c70820e773bac96c907287e860d8b9d98bc68778d3880fb1885` |
| `outputs/qa/reset_multitask_adaptive_matrix_20261113/result.json` | `b9e66fd81420049508f921644df3c5cbc0e25d19069e2d59c16527e8142a0cca` |
| `outputs/qa/reset_multitask_adaptive_preflight_20261113/result.json` | `02d109a557be4004f372648d9ba8a21c8c0dc69890b6f47786158a52b4dd6114` |
| `outputs/qa/reset_multitask_adaptive_preflight_v2_20261113/result.json` | `4937f4128f99b9f56124617623d2a64155a1c32f2acfb7c9eda716e32b566415` |

新入口在实际模型 Python 环境导入时未提前加载 OpenCV、PyTorch 或 Cosmos；CPU
解释器中的 checker 导入隔离通过。这些是软件和来源准备，**尚未执行本矩阵的实际
模型、物理或独立重放**。最终 64 并发相关回归、零模型 dry 和实际共享预检仍待完成；
原正确参考正控及最终 24 条留出验收不由本阶段替代。
