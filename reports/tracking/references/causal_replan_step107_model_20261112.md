# 因果重规划边界的 step107 模型参考

2026-11-12。按[登记方案](../../../plans/v0.0.1/tracking/references/causal_replan_step107_model_20261112.md)，
对[已重放核验的 step107 当前图像](causal_replan_step107_capture_20261112.md)执行一次
真实 WAM 请求。请求、响应与实际图像/状态 SHA 一致，固定 `8400_masked`、seed 42、
H40、前十维 Action128 mask 和 35 次去噪。模型响应前八个 opening 为
`0.123116, 0.102093, 0.090949, 0.115550, 0.097430, 0.094162, 0.084572, 0.089142`；
它们是期望参考，不是夹爪已执行状态或任务结果。

实际模型初始化 1 次、请求尝试/完成 1/1、去噪 35 次，模型请求耗时约 13.34 s；
请求及模型 worker 均实际 wait/reap、exit 0、未发信号。指定 GPU UUID
`GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf` 推理前后均为空闲，进程结束后已释放。
32 并发 CPU 回归 40 项通过、0 skip；零模型 dry 与 1004 个源码文件、163 个
权重文件的前置检查通过。

本轮没有物理积分或 fresh case；采图时的仿真器已经关闭，因此这是离线生成的
当前图像参考，不构成实时闭环。累计同系列物理成本保持 1,305 周期、32,625 内步、
72 个仿真模型/context及另外 2 个静态模型；世界模型初始化 9 次、请求尝试/完成
21/18 次及去噪 630 次。新参考尚未用于派发动作，需要重建 107 周期因果历史、
在原质量和安全门下执行并独立重放；本结果不证明任务、泛化或真机成功。

| 仓库内证据 | SHA-256 |
| --- | --- |
| `outputs/causal_replan_step107_model_20261112/result.json` | `46dc2c1d1ce1d5150ccede4cad68f695e535eba15b8a10fbebb9ff304e4302d0` |
| `outputs/causal_replan_step107_model_20261112/producer/model/model_run/step107/response.json` | `38586ec47073c2b0c7fbcf94fbf9dce6ce59d76fadc6704bdfa78d4074ea7aca` |
| `outputs/causal_replan_step107_model_20261112/producer/simulation/response.json` | `0840f54270785f81980b31a266d9212cc430c64651790a2cfd2492aa3974288a` |
| `outputs/causal_replan_step107_model_20261112/preflight.json` | `003ff0ae7adcab964b6c190dddb58ee730c9f03dbfb2c8a97a1f17b8af279143` |
| `outputs/causal_replan_step107_model_dry_20261112/result.json` | `00fbf63d6c255b9e17ab2304a83d9b1c5f9fbdac80f0fce191d8990039ef6771` |
| `outputs/qa/causal_replan_step107_model_20261112/result.json` | `3bb74e166fba8e5627d2a398998ccf6b9b440570ed3d9614d3a8adb54c010bc3` |
