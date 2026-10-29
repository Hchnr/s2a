# 迁移运行时的 CPU OSMesa bootstrap

2026-10-29：实现已提交 `5c2c874`，统一 native QA **167 passed / 1 skipped**，耗时136.71秒。实际 Ruff、format、CLI dry 准入已由原会话12415正常exit0关闭。唯一真实bootstrap在原会话78245以exit1关闭：**运行时导入成功，首次模型构建前失败**。`runtime_loaded=True`，`bootstrap_verified/control_qualified=False`；没有成功模型、初态恢复或渲染结果。

本项落实[登记计划](../plans/v0.0.1/native_runtime_bootstrap_20261029.md)与[固定配置](../configs/native_runtime_bootstrap_20261029.json)，补齐[文件快照](native_runtime_path_snapshot_20261029.md)之后的实际加载证据。它与已完整关闭的[H12数值复算](restored_anchor_sqp_20261029.md)独立，不授予控制、物理pilot或任务资格，GPU继续禁用。

## 固定范围与实现

输入仅使用已闭合快照、独立v2文件核验及其实际wait/reap记录、原runtime manifest中的record、原controller合同和当前依赖。只恢复libero_goal/task8、seed42、trial0的79维原初态，不读取HDF、训练dataset、expert动作，不执行actual368前缀、S2A动作或新世界模型参考。CLI在实际环境加载前绑定全部来源及QA，结束时重核；见[输入准入](../scripts/verify_native_runtime_bootstrap.py#L148)（`scripts/verify_native_runtime_bootstrap.py:148`）。

预算是一次环境构建，最多3个模型、3个OSMesa context、32次forward、2次step1、一次reset及一次256×256 agentview渲染/read_pixels；control step、mj_step、mj_step2预算均为0。静态预计16次forward是预期值，实际次数待运行记录，不能作为已执行计数。step1单独记账并检查time不推进。见[预算](../src/s2a/native_runtime_bootstrap.py#L24)与[实际调用观察](../src/s2a/native_runtime_bootstrap.py#L249)。

保持冻结成功路径的配置和控制器，唯一reset接口差异是直接调用一次`env.env.reset()`，避开外层吞异常及自动重试。恢复后的state要求精确一致；渲染前后state/time字节保持，实际renderer须为CPU软件OSMesa，图像为非恒定uint8 RGB。见[reset](../src/s2a/native_runtime_bootstrap.py#L491)、[初态检查](../src/s2a/native_runtime_bootstrap.py#L522)、[图像检查](../src/s2a/native_runtime_bootstrap.py#L541)。

执行前检查和合成测试已修正：原机器人名应为`MountedPanda`；允许原XML的相对`meshdir`但实际file必须迁移到批准的绝对路径；部分环境构造失败仍尝试一次cleanup；临时文件的`dir_fd`写入检查按实际目录解析。Python audit写入守卫限制到新work目录，但不等同OS syscall沙箱。CLI失败保存实际阶段、计数和原异常；若归档又失败，只向本函数原异常附注发布错误，不用调用者外层异常推断本次状态。无原异常时发布失败正常传播。见[cleanup/文件守卫](../src/s2a/native_runtime_bootstrap.py#L294)与[CLI异常保留](../scripts/verify_native_runtime_bootstrap.py#L428)。这些是执行前修正，不是已经发生的真实bootstrap失败。

## 已验证证据与待执行项

统一QA绑定17份最终source/plan/config及5个测试模块，使用Python3.11.15、空CUDA/LD、单线程、禁bytecode和外部pytest插件。子进程PID938526，原会话43660实际exit0；receipt的`actual_wait/reaped/complete/passed=True`、`returncode=0`、`signal_sent/automatic_retry=False`。已再次逐项核对17份源码、QA日志和driver SHA，`source_unchanged/driver_unchanged=True`。

唯一skip来自[旧环境测试](../tests/test_native_environment.py#L112)：历史import-only probe fixture未安装，因此该旧cv2导入测试未执行，不能计为通过。本次真实bootstrap的独立cv2准入已成功：已知导入LD变更符合登记probe，恢复后CUDA/LD均为空，loader/config来源前后一致。

| 证据 | SHA256 |
| --- | --- |
| [QA receipt](../outputs/native_runtime_bootstrap_qa_20261029_642s15ix/result.json) | `e0630479475f117174ed3823d65017f6195e79806ec6079ad6ed1d45bc7b0f62` |
| [QA日志](../outputs/native_runtime_bootstrap_qa_20261029_642s15ix/pytest.log) | `e50d18219257ebe01d0be6901a661f0f56912202d021681cc46b1a528d96c173` |
| [QA driver](../outputs/native_runtime_bootstrap_qa_driver_20261029.py) | `5e2968c83460e71f33f43b351b66fc9e0ea1d86f25f72c76ce647b4d78f389ff` |
| [准入/执行driver](../outputs/native_runtime_bootstrap_execution_driver_20261029.py) | `17bbf0657075fe73549db9ef4b691af7b09903399962939cdc905eed0666eea3` |
| [实际准入receipt](../outputs/native_runtime_bootstrap_admission_20261029/result.json) | `aea519230e66a2abf4fa4196885d1d81cb51e7de2c8a1608340e55b3db725fdc` |
| [失败bootstrap receipt](../outputs/native_runtime_bootstrap_20261029/result.json) | `c4e1742a0480a9a02dca19f6c65fab8f686c7c9c48e300fc634b288159da90b4` |
| [失败owner receipt](../outputs/native_runtime_bootstrap_execution_20261029/result.json) | `98e3a7d514d0abc76c79aa5dda50bf380831aa219657d72d1fc4798577a7892a` |
| [实际bootstrap日志](../outputs/native_runtime_bootstrap_execution_20261029/00.log) | `f20fd83670380e07c51e22429e2f1738dfbfb7ba2648c50ee43cc706fb730e33` |

实际准入的3个子进程PID946839/946932/947034均实际wait/reap0，无signal/retry；同一QA与driver绑定，21份登记来源及3份日志重核一致。正式CLI dry检查2352个来源引用，明确environment_initializations/new_responses/physics_steps=0、runtime_loaded/bootstrap_verified/control_qualified=False。

另一次开发期只读文件预检已实际exit0，核对2337个引用、45份历史LIBERO源码与20个当前依赖，53.045秒；当时`cv2/mujoco/libero`均未导入。此记录不替代正式CLI dry或真实bootstrap。

## 实际失败与后续边界

子进程PID952611、父PID952544；owner实际`wait/reap=True`、`returncode=1`，无signal、无自动retry。CLI与owner均保留`complete/passed=False`。CLI结束来源重核登记2669个引用，`source_unchanged=True`；失败未改写快照或执行源码。

失败阶段是`constructor`，调用链为`compile_model → _asset_records → _safe_path`，在底层模型编译前拒绝：

```text
.../LIBERO/libero/libero/assets/scenes/../textures/light-gray-floor-tile.png
```

这是资产路径语法与守卫的兼容问题：守卫在[源码第93行](../src/s2a/native_runtime_bootstrap.py#L93)无条件拒绝`..`；该引用规约后仍在批准的assets根内，文件位于快照inventory中，SHA `a2aae4ba3b38c3ab9851eca9ad6e76cef9357ef75b3e9a4b280f694ca30acbc5`，已只读核对存在且匹配。不能据此推断其后的模型构建或渲染一定成功。

| 实际操作 | attempted / completed |
| --- | --- |
| configure | 1 / 1 |
| model_construct | 1 / 0 |
| context、forward、step1、reset、render、read_pixels | 全部 0 / 0 |
| control_step、mj_step、mj_step2 | 全部 0 / 0 |

`model_records=[]`，无成功模型或任何forward/step1。局部对象清理已尝试一次，`cleanup_target=inner_partial`，但该对象尚无`sim`属性，记录`AttributeError`且`cleanup_complete=False`；原资产路径`ValueError`及其`BootstrapFailure`异常链仍被保留，清理异常没有覆盖主错误。

下一步[隔离v2修复](../plans/v0.0.1/native_runtime_bootstrap_v2_20261029.md)已登记：在仍拒绝越界、symlink和旧根路径的前提下，验证批准根内合法路径规约，并单独处理尚未生成sim的局部对象清理。保留本次失败产物和冻结源码，使用新输出与新的QA/准入，不能原地修改后自动重跑。实际渲染、79维恢复和控制器核验仍未到达；原H12 witness、原任务门槛及总目标保持，控制资格仍为False。
