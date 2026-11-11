# GPU 恢复后的世界模型单次复现

2026-11-11。用户确认当前容器可见四张卡均可用后，按[登记](../../../plans/v0.0.1/tracking/references/gpu_resume_repro_20261111.md)
在 `GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf` 上运行。实际设备为 NVIDIA H100
80GB，PyTorch `2.10.0+cu130`。先完成 GPU 矩阵计算，再加载既有 checkpoint
`iter_000008400`，执行一次 task0/init0/seed42 原始开发输入的 H40、35 次去噪推理。

独立读取新旧响应核对：两者各 40×128 的 raw action 和 normalized action 均逐元素
相等，最大绝对差和不同元素数均为零；后 118 维为零。该比较针对数组及请求的固定
关键参数，不宣称整个响应文件逐字节相同。准入核验了固定 Cosmos 快照中 1004 个
源码文件、163 个权重文件及图像、请求和历史响应的 SHA256；全部符合登记。

GPU 工作者实际退出码为 0，wait 后观察到对应设备显存 0 MiB、无计算进程。CPU
专项检查 8 项通过、0 跳过，全仓 Ruff 和格式检查通过。依赖初次加载受共享存储读
取延迟影响；期间的并行依赖预读只读环境文件。本次无训练、新初态、物理步或自动
重试，不能从该复现推导闭环性能、实时性或项目验收结果。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/gpu_resume_repro_20261111/preflight.json` | `52e3f7c7e3aff23285b3f56e871afb1fb008aaefeb9ba8b2ef8c3d2d2db1b34a` |
| `outputs/gpu_resume_repro_20261111/gpu_smoke.json` | `b803f1158309753f3109f2da893cff838845d9c9dd9c41a3dd503333f8034688` |
| `outputs/gpu_resume_repro_20261111/model_run/result.json` | `0515e706ac5f59d8a0ddafb5f15b291e597f4fc6e83e53c7e5b9be152106819b` |
| `outputs/gpu_resume_repro_20261111/model_run/response.json` | `9669f30418803ac1f074b47d1fe9b132642fac2a4a0a37bd2509d2c603e9b45a` |
| `outputs/gpu_resume_repro_20261111/gpu_after_wait.json` | `644ccc617059f1e653a7e752c0cce9a8796905c0e92092c64c0c4bdfef2b689a` |
| `outputs/qa/gpu_resume_repro_20261111/result.json` | `fea180a35285b2862b83ae1d38ea927e37f07d787b1571f7813f2bcd02f7152b` |
| `outputs/tracking_preview_wam_development_v5_20261028/development/service/requests/000_task0_init0_s42_raw_q00/response.json` | `c057826480052aa7f90b53ddeaac936709112609a42bfcdee107bb42d1a082a9` |
