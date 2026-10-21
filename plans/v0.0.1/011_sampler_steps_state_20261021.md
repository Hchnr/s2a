# 8400采样步数与实际噪声配对诊断 — 2026-10-21（Asia/Shanghai）

状态：实施前冻结，独立state诊断，不改变010的35步闭环协议，不打开新初态或自动增加物理预算。

现有8400为H8/H16三指标较好的候选；35步/shift5是当前统一推理设置，没有已执行的同输入70/140步配对证据。当前源码审查没有确认采样链路错接；增加步数是待检验的生成质量假设，不预设更多步骤更好。延迟不在本轮验收范围内。

固定8400非EMA、active十槽mask、guidance1、shift5、H40/20Hz、task-only、原24诊断case/6旧episode/22独特anchor、seed42/43。逐步数35/70/140各48次，共144个新H40请求、0物理步；重复phase按既有协议保留，另注明44独特anchor-seed。所有输入PNG/数组及归一化沿用原始manifest，不重新选择样本。只用物理GPU6/port9566的单私有模型、单rank/worker且串行，CPU编排CUDA和LD为空。

新增隔离serving/trace/coordinator入口，不改任何冻结执行/服务/审计实现或外部Cosmos文件。复用现有load_service与active-mask，先安装mask后安装只读trace。保持原sampler对象，包装实例forward与其velocity callable，原样调用和返回；不使用velocity_postprocess_builder、net hook或新调度器。实际sampler入口tensor保存dtype/shape/stride/device/storage offset和原始value bytes SHA（bf16用uint8视图，不能float转换）；这是conditioning/mask之后的实际初态，不称mask之前pure Gaussian epsilon。

实测velocity调用次数和传入integer timestep；实际sigma/scheduler通过受源码SHA及code identity保护的原forward caller-frame只读取得，未知结构立即失败。参数identity/storage/version、eval/backend/compile/reuse条件、输入/条件及源SHA前后绑定；不修改tensor或RNG、不新增网络求值。网络forward次数仅在冻结源码支持下标为derived，不冒充直接hook计数。实际noise SHA对比是跨步数配对准入，不能只凭seed认定配对。

先完成全部48个traced35，并与原8400_g10同输入cached35的full40×128 raw/normalized逐值exact核验；任一不一致就停止，保留故障，不执行70/140，不放宽精度。准入后按原顺序执行70、140各48次，逐对核入口noise/conditioning相同与实际NFE/schedules，缺失或不一致停止并保留，不自动重试或替换样本。原35没有noise witness，exact输出只认证这些输入的trace回归；新35/70/140有实际初态证据。

沿用完整权重指纹的path/size/mtime检查，不重复读取旧32GB。新入口先CPU meaningful测试（原样转发、无RNG/对象变更、原始bytes、请求隔离、guard fail-closed）和独立代码review，再启动模型；仅停止自身PID/startticks绑定的进程组，GPU/端口占用则停止。输入、运行配置、源、预算与launch commit和完整release均保存，禁止覆盖。

全部完整后独立从raw response积分state，报告H8/H16/H40位置/SO3/opening及逐episode/phase、hold/constant velocity；保留1778/384/768有效端点口径，另明示重复anchors。70/140若没有有用state优势，不追加对应物理；若有则另固定候选和物理预算，不中途替换010。本轮不训练校正器、重加task规划或改变mask/归一化/旋转合同。


## 运行前固定的后续候选门槛

本state筛查的主要用途是短重观测，因此以H8/H16六项指标为候选门槛：相对本次traced35，位置/SO3/opening六项均不退化（比较数值仅容1e-10绝对误差），且H8或H16位置RMSE至少改善10%，才允许另登记对应更多步数的物理开发候选。H40全部三项及逐episode结果仍完整报告，不能只报前缀或成功phase。若70和140都合格且互有取舍，先固定70作为较小变更；若140六项均不差70且至少一项位置再改善10%，则可选140。没有合格项就不追加更多步的物理；本条在任何011模型请求前补充冻结，不修改144调用和010预算，也不允许据结果调门槛。
