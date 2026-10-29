# CPU bootstrap v3：补齐历史record资产的路径准入

2026-10-29。承接[v2实际失败](../../reports/native_runtime_bootstrap_v2_20261029.md)：根内资产规范化已让constructor/hard_reset成功，实际2模型、2contexts、9forward、2次非积分step1，随后record_xml前置守卫拒绝`..`。原record的99个file属性中实际有4条历史`/Users/.../assets/scenes/../textures/`引用，先前路径调查漏查了它们。环境已成功关闭，mj_resetData/render/控制/物理步均0（高层hard_reset已完成）；2753来源重核不变。v1与v2失败及旧23源全部保留，不原地修补执行来源。

本轮仅修正这个已定位的不一致：记录声明仍须为绝对路径；旧外部路径沿原`/assets/`suffix作字符串重映射，不stat或读取历史根；含`..`的资产最终交给v2已有的逐组件规范化验证，必须始终留在批准根内且无symlink、失效中间组件或逃逸。未知外部路径仍拒绝。完整最终XML在模型编译前经过这一守卫，原XML非file结构与数值、snapshot、venv均不改变；不能因延迟检查而跳过检查。

新core仅在v2私有函数AST上替换唯一record路径前置条件，并逆比较确认其它AST不变；复用v2的资产记录及局部清理实现。证明明确标记record前置检查已改变，不沿用v2的“未改变”标签。新CLI在私有namespace中绑定新版本路径与core；保留完整旧23来源、v1父失败和v2直接前驱失败。只读核对实际wait/reap1、无signal/自动retry、计数、time不推进及outer cleanup成功，不能将前驱改记为通过。

[新配置](../../configs/native_runtime_bootstrap_v3_20261029.json)在v2基础上仅修改output_dir并新增repair_predecessor五引用；repair_parent的v1记录仍原样保留。固定任务/seed42/trial0、完整初态、控制器、20Hz/2ms、3模型/3contexts、forward上限32、step1上限2、reset1及单帧软件OSMesa预算均沿用[v1](native_runtime_bootstrap_20261029.md)与[v2](native_runtime_bootstrap_v2_20261029.md)。不延长600秒child上限，仍禁止control/mj_step/mj_step2，GPU禁用、空CUDA/LD与单线程保持。

统一QA绑定旧23加新core/CLI/两测试/本计划/配置共29源，执行原7加新2共9模块。新增测试直接覆盖四条真实历史引用的纯文件映射、合法恢复路径和越界/symlink反例，以及前驱准入和有限AST证明；不加载真实模型。实际Ruff/format/CLI dry通过后，仅在 `outputs/native_runtime_bootstrap_v3_20261029` 执行一次，保留实际退出、计数、源码及资产SHA，不覆盖失败或自动补跑。

成功仍必须完成原79维初态字节恢复、固定控制器检查、真实软件OSMesa非恒定256RGB帧、渲染不改变state/time及全部来源结束重核。仅证明当前运行时bootstrap，不证明历史等价或控制成功。原完整H12 witness0、实际cursor8/task/grasp0及18/12/24任务门槛保持，不由本项启动物理pilot。
