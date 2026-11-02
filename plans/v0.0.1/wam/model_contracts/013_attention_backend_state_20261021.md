# 8400训练注意力后端的原生推理对照 — 2026-10-21（Asia/Shanghai）

状态：模型请求前冻结，新增48state/0physics条件预算；不改变010/011/012，fresh性能保持未开。

8400保存训练配置是action_tower_attention_backend=flash3_video，S2A既有load_service显式覆盖sdpa。之前三个任务×三个sigma的同权重训练前向在恢复训练backend及固定padding后，相对SDPA最大有效velocity差.0234375、最大RMS.00551005；它小于已修的inactive-noise效应，但不是完整35步生成的影响证明。故此处只比较原生推理的attention backend，不能预设这是失败原因或因速度更快而宣称质量更好。

固定8400非EMA、原24cases/6ep/22anchors、seed42/43、task-only/H40/20Hz/guidance1/35steps/shift5、mask/归一化/相机及推理padding策略（disabled）不变，只把实际attention backend从SDPA换为保存的flash3_video。不同时恢复训练fixed padding，不引入任务规划/改标签或controller；这不是完整历史训练数值重放。

仅物理GPU7/port9587单模型单worker串行48H40请求、0物理。启动条件：012全部48完成独立核验并正常释放GPU7；011全部48traced35输出raw+normalized与旧缓存exact、其admission35及noise/condition trace完整；实际home Cosmos树backend能力/接口和新隔离代码CPU测试及独立review通过。任何条件不满足不启动，不自动重试/替换卡或预算。011的70/140仍在GPU6按原协议继续，不等它们挑结果。

新增独立backend加载/serving/coordinator入口，显式复用原构建逻辑但不编辑冻结wam_runtime/旧serve/011trace或外部文件。保留model/decoder的--no-compile、no-small-action-metadata、active-slot-mask、单rank/DP/CP/CFGP=1与同checkpoint加载。新代码完整SHA、原构建逻辑差异、实际加载source、配置与权重stat绑定；不重hash旧32GB。只停止自己PID/startticks绑定进程组，拒占用和覆盖。

复用冻结SamplerTrace观察实际sampler输入、clean latent、mask/reference/token/caption/image、NFE/integer timestep/actualsigma，在第一次velocity求值前逐输入与011traced35 pair_identity exact核验。参数对象在各服务内部前后不变；跨服务权重身份由同DCP完整指纹和显式仅backend override支持，不把两个进程不同pointer hash硬比较。实际branch/reuse flags分别保留，任何非backend输入不一致立即停止、保留失败，不放宽。

完整48后独立raw积分H8/H16/H40 GTstate、hold/恒速及逐episode/phase，并报告与SDPA同配对生成state距离。候选门槛同011：H8/H16六项均不退化（绝对数值容1e-10），至少一个位置RMSE降10%；不满足不追加backend物理。即使满足也只允许另预注册完整多任务开发，不自动替换010或打开fresh。不把单次前向/更快/NFE相同当作任务成功或训练正确。


启动前源码澄清：actualhome的flash3_video是hybrid backend，VIDEO用FA3、UND/ACTION用Flex；Flex helper在CUDA下内部torch.compile(flex_attention,dynamic=False)。本协议只保持model/decoder compile=false，不声称所有kernel都无内部编译，不额外禁用Flex内部compile。此为固定backend选项的实现差异，实际reuse flags/NFE也记录。编译cache写本实验outputs下；不从更快、编译发生或单kernel差异推导任务效果。该澄清在任何013模型请求前固定。
