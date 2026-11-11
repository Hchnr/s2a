# 提前干预后两张当前图像的 WAM 推理

2026-11-11。此前[两个物理分支](early_tabletop_capture_20261111.md)从相同
step118 状态各执行两周期，并分别在 step120 采集经独立重放核验的当前图像。
两张图像及状态 SHA 不同；按[固定计划](../../../plans/v0.0.1/tracking/references/early_tabletop_intervention_live_wam_20261111.md)
对每张图像分别运行冻结 WAM checkpoint iter_000008400，公开任务相同、seed 42、
H40、每次 35 步 denoise、Action128 仅前 10 维有效。模型输入不含未来实测状态
或任务标签。两次都在同一模型进程中执行，输出路径和图像/状态绑定分别保存。

| 分支 | 图像 SHA256 | 实际 denoise | 模型请求耗时 | 响应 SHA256 |
| --- | --- | ---: | ---: | --- |
| 原目标 | `3325714b3cf8f65c2f45d59996d3585820d95c2390cb336a1fd414200cdac45a` | 35 | 13.236 s | `84e034b51a2f88668a26e3174e4d2e242901b187d344b00d3e455a43b98d21b6` |
| 位姿包络投影 | `1a2a6a58ada903a641972144bb3084fe5d6a78b94afa064026f4bee20a8cd145` | 35 | 11.714 s | `a0bf4e0a86635809f9fbc4e1f253e08b74fd96e13997924b23c2d63e0b2dbf43` |

两份输出的 400 个有效标量均不同；其位置、旋转表示和夹爪通道语义不同，不能
把混合维度的差值当作物理误差或任务质量。输出仅证明各自当前图像得到了独立
生成的参考，尚未执行后续物理反馈或独立评估预测效果。

运行前 CPU 专项 44 项通过，源码与 163 个权重文件通过 SHA 检查；显式选用已
授权的 `GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf`，一次模型初始化、两次
请求、70 次实际 denoise。GPU worker 实际 wait/reap、退出码 0，运行前后该卡
显存均为 0 MiB。模型期间仿真暂停，上述秒级耗时不满足 20 Hz 实时闭环要求；
零新增物理步、零新初态，不授予控制或任务验收资格。

下一阶段须重新物理重放各自 step118–119 动作，验证当前状态及图像与模型请求
一致后，分别执行各自新生成参考的最多八个周期，并做不加载候选策略的独立
checker。原 18/12/24 留出与每条 600 周期稳定成功标准不变。

| 仓库内证据 | SHA256 |
| --- | --- |
| [模型零工作 dry](../../../outputs/early_tabletop_live_wam_model_dry_20261111/result.json) | `15849ff1651c0ac9cadce4074ffad29698b6300193388042076116c8d908a1a6` |
| [模型 32 并发专项](../../../outputs/qa/early_tabletop_live_wam_model_20261111/result.json) | `bfa8ae5114c0e07adc17e6145cdabca1d44651f740faf3f01411b3979d8508f1` |
| [源码与权重 preflight](../../../outputs/early_tabletop_intervention_live_wam_20261111/model/preflight.json) | `8b7a245a2a23f0b21eba607225c2f4e3e65320b383caf93eebbee6884c6a6cde` |
| [GPU 设备 preflight](../../../outputs/early_tabletop_intervention_live_wam_20261111/model/device_preflight.json) | `d7a064fd1f8ab0e67eb2f6a03f602b1ae8f95b4befbe8005f6bd418e330a73b8` |
| [模型 owner](../../../outputs/early_tabletop_intervention_live_wam_20261111/model/owner/result.json) | `0dea3e748ad134a0bae785944af8f69fb25e4c0deda7b6ed397700424e8802d9` |
| [两次推理与去噪记录](../../../outputs/early_tabletop_intervention_live_wam_20261111/model/model_run/result.json) | `46cb9f6c1b1b0e20cd284969c69adef5bc445da37dbccd32f110eda8a7512724` |
| [原目标分支响应](../../../outputs/early_tabletop_intervention_live_wam_20261111/model/model_run/original/step120/response.json) | `84e034b51a2f88668a26e3174e4d2e242901b187d344b00d3e455a43b98d21b6` |
| [位姿包络分支响应](../../../outputs/early_tabletop_intervention_live_wam_20261111/model/model_run/pose_bounded/step120/response.json) | `a0bf4e0a86635809f9fbc4e1f253e08b74fd96e13997924b23c2d63e0b2dbf43` |
