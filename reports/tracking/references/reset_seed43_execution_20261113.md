# seed 43 复位闭环的进程监督

2026-11-13。已将预登记的 seed 43 复位仿真、实际当前图像模型请求和独立重放
接入同一进程监督入口。producer 绑定显式 GPU UUID、独占输出和预算；checker
只有在 producer、模型及 GPU 关闭并复核输入、worker 生命周期、实际响应和
累计开销后才能启动。中断或缺少收据时保留部分或有界未知开销，不自动重试。

使用历史保存的物理、模型和 checker 收据进行接线回归，验证了 16 个周期的
结果汇总、一次模型响应、独立重放一致性、GPU 未释放时拒绝重放，以及中断和
更改预算时的处理。这些是测试替身和已有收据，**不是本次 seed 43 现场结果**。
新入口在独立 CPU 解释器中通过 checker 策略模块和仿真模块导入隔离检查。

32 并发相关回归 39 项通过、无跳过或收集错误，worker 均已等待回收：
`outputs/qa/reset_seed43_execution_v2_20261113/result.json`，SHA-256
`619bff1dbaee05f1167ce5fd026e25d6d6fba31dddcd30e3b05462d40cf4a0c5`。
修改文件的 Ruff 检查与格式检查通过。首次回归使用 `/opt/venv`，其中缺少
pytest，因此 0 项收集并失败，保留于
`outputs/qa/reset_seed43_execution_20261113/result.json`，SHA-256
`3ccd2c1d0107709e425c917ec67eb0b4cb39a7e863faed0cc6fb9a0c666db7a1`；
通过的回归使用仓库内 `outputs/libero_env311` 环境。

尚未执行本试验的 64 并发最终相关回归、dry、模型/环境预检、现场 producer
及独立 checker；原三任务、600 周期和留出验收不受这些接线检查替代。
