# 三任务开发初态的现场复位和首图准入

2026-11-11。按[现场方案](../../../plans/v0.0.1/tracking/joint_control/reset_native_admission_20261111.md)，
在 CPU/OSMesa 上为 12 条已审核 v4 开发案例分别创建环境、恢复其记录的 step0 状态，
核对固定控制器、现场端点、夹爪 motor 和原首个 WAM 请求图像。12 条案例只有三个
开发初态，各自两 seed、两个参考变体；本阶段没有派发控制命令、推进物理或调用 WAM。

## 首次执行：状态一致，图像不一致

12 个 worker 的现场状态 SHA 均与原记录逐字节一致，逐项状态/控制器/motor 字段
最大差为 0，渲染前后状态和时间不变；但重新生成的 12 张 canvas RGB 哈希均与原
WAM 请求不同。原 owner `complete=false/passed=false`，12 个 worker 均 exit1、
实际 wait/reap 且无信号；均已清理环境，动作派发前停止。实际开销为 **36 个模型、
36 个 OSMesa context、0 控制周期、0 物理步**。原失败及各 worker、图像和生命周期
收据原样保留，不能重分类为成功。

原请求 PNG 由既有独立审核逐 SHA 固定。针对现场已保存的两路 256×256 相机数据，
使用相同 PyTorch resize、裁剪和拼接，在独立 CPU 进程重建单线程及两线程 canvas：
12 条单线程结果逐像素复现本次现场 canvas，12 条两线程结果逐像素复现原请求。
差异在三个相关初态各自完全相同，所有变化通道值均处于用整数有理数重建的双线性
插值精确半整数位置：

| 开发初态 | 不同通道值 / 1,843,200 | 不同像素 / 614,400 | 最大灰度差 | 位于精确半整数的不同通道值 |
| --- | ---: | ---: | ---: | ---: |
| task0 | 1,680 | 1,559 | 1 | 1,680 |
| task1 | 1,779 | 1,647 | 1 | 1,779 |
| task2 | 1,696 | 1,564 | 1 | 1,696 |

两线程被选为对已查看数据的修复，非预先独立选择；此结果表明线程数会影响这批
图像的取整，不声称底层唯一机制，也不证明其他硬件或场景均能复现。最初在尚未
启动模型的准备探索中曾遇到模块引用错误和提前导入 MuJoCo 导致的 OSMesa 导入
错误，均在正式预算前修正；轻量加载器的测试现在要求配置仿真前不导入仿真模块。

## 独立预算的两线程修复

按[修复方案](../../../plans/v0.0.1/tracking/joint_control/reset_canvas_thread_repair_20261111.md)，
保留所有数值和图像精确门，只在原 `render_anchor` 调用期间将 PyTorch intra-op
线程从 1 临时设为 2，成功或异常后恢复至 1；worker 的数值库/OSMesa 环境线程变量
仍为 1。新目录的 12 个现场 reset 全部通过：状态字节、各比较字段和现场 RGB 图像
均与各自原记录一致；线程范围均记录为 `1 → 2 → 1`，12 个进程 exit0、实际
wait/reap、无信号，环境均清理。没有历史命令被执行，`previous_saved_command`
为 `null`，旧的“已回放历史动作”标志明确不适用。

本轮独立结果审计重新读取 12 份现场原始相机、现场 PNG 和原请求 PNG，逐像素
验证两线程重建及原/现场图像一致，同时检查状态哈希、零步预算、线程恢复和进程
收据。它是对保存产物的只读复核，**不是另一轮现场物理重放**。

新阶段使用 36 模型/36 context、0 控制周期/物理步/WAM/新初态；与原失败累计
**72 模型/72 context、0 控制周期/物理步/WAM/新初态**。原失败图像审计及新结果
审计各为 0 模型、0 forward、0 物理步。此前从三任务现场原相机探索不同线程数也只
执行 CPU 图像运算，未创建模型或推进仿真。

运行前专项：原准入 32 并发 40 项通过；图像审计 17 项、修复准入 37 项、结果审计
14 项均以 32 并发通过、零跳过及收集错误。全仓 Ruff/格式检查通过；未重跑完整
pytest，先前仿真环境全量的 4 项失败、2 项跳过及 5 个文件收集错误依旧保留。
此阶段尚未执行 8 周期控制、独立动作重放或新图像 WAM 闭环，不能声称任务成功；
原 18 条正确参考、12 条 WAM 开发、24 条最终留出及稳定成功门槛不变。

| 仓库内证据 | SHA256 |
| --- | --- |
| [原准入 32 并发专项](../../../outputs/qa/reset_native_admission_20261111/result.json) | `1de4c82421f99b131c69bd9e8b0800af2d7da157509085f46ca13499e67a9888` |
| [原零模型 dry](../../../outputs/reset_native_admission_dry_20261111/result.json) | `0c212f073fd79c93bbecf86953553011f2a49e9cd76978134b1432e5058f9f21` |
| [原现场失败](../../../outputs/reset_native_admission_20261111/result.json) | `96e3732026a52470348b51a4aeabe001e01adc9261a8ab728f18564c945310a0` |
| [原 12 worker 生命周期](../../../outputs/reset_native_admission_20261111/workers.json) | `caa8f1d4bc1c9d0240cedd0d570314855d0d3b2b065f9ee4b7a5aab5e0b98167` |
| [原图像线程审计](../../../outputs/reset_canvas_threads_audit_20261111/result.json) | `e433f00f1b22970f6720116fdf736e691de98073efb69907c0e0c72759581268` |
| [修复 32 并发专项](../../../outputs/qa/reset_native_admission_v2_20261111/result.json) | `3e6a7aeb6e367f233b79cd09084f873e9f8a1864f004f654d546e4e7ab98ebb2` |
| [修复零模型 dry](../../../outputs/reset_native_admission_v2_dry_20261111/result.json) | `98c1d96fdb3d99a7bb7d7053c4758f84376087cb0c15755ce50bf7361a75b135` |
| [修复现场结果](../../../outputs/reset_native_admission_v2_20261111/result.json) | `b6fc7fd1bd62485297e2b33ad9299a9c682cf4df0cd95da3ec58ac264539d411` |
| [修复 12 worker 生命周期](../../../outputs/reset_native_admission_v2_20261111/workers.json) | `b0c4489dd8997a51f6e6adec6ce13be9282d47c32705eb1bd5e5b4bd52f7fe6f` |
| [独立结果审计](../../../outputs/reset_native_admission_v2_audit_20261111/result.json) | `6cb975bb51f2641122423e31c8cd7eac7c81c553d77f6525e9abae5e0d2385a2` |

执行 QA 绑定的更新前 `docs/status.md` SHA256 为
`c1ca1a7818bc81bbfe473b2eb8fb43475286b4f9076b3ac71afa30e284771512`；
原字节见[历史副本](../../../docs/history/experiments/status_before_reset_native_7c309b3a.md.txt)。
