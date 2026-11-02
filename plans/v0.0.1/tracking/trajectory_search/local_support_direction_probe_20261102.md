# 终点支持方向的固定非线性核查

2026-11-02。前序48LP提示终点旋转的局部模型瓶颈，但方向步长接近.2，
没有非线性误差界；不把局部下界用作物理不可达证书。

只选已关闭local_tracking_support_20261102的lp_48.json：small-progress最终
anchor23、终点rotation group23、reserve .9。原文件SHA固定
6ff37e4cf90c1763cdffe0c961c6078c8838371351909f9b38b1470a2630b09b。
读取保存point，不重新求LP或拟合。先独立重建全部4369安全/命令/rate行，审核
原单位primal/dual、box残差修正和固定模型的完整预测。

固定5条完整H12命令：anchor、x+1*d、x+.5*d、x+.25*d、anchor repeat。
保留原root global域、输入范围和.25 rate，不裁剪、不重采样，不按结果换方向。
这是诊断，原生产trust radius和pose ceiling不作为诊断响应准入，候选不下发；
实际完整joint/buffer、reserve、24caps和全部错误仍逐项记录，失败不隐藏。

生产每条均从原actual368/历史367状态重置，使用相同replay-order预测器；
独立native model完整重建5条，沿原1e-10/torque1e-8及布尔审核，响应/metric
逐值exact，初始历史与repeat exact。记录固定anchor法向的实际支持、线性支持、
终点rotation norm及全部时刻误差、safety预测残差、caps/prefix/witness。
结果只用于判定当前模型的跨域可信度和下一步路径方法，不授予控制资格。

新预算5生产+5独立=10响应/3000内步/3122查询/2model，0新LP/QP/NNLS/
stencil拟合/物理/GPU/WAM/fresh/最终留出/实际参考消费；新独占
outputs/local_support_direction_probe_20261102。执行600秒TERM/10秒KILL，
实际wait/reap，失败封存不自动重试。先实现和32并发QA、固定来源与代码指纹，
再执行，期间不修改source/config/docs。CPU空CUDA/LD、数值库线程1。

若实际支持明显偏离，优先处理模型外推和非线性路径，不能继续拿LP下界推断
真实不可达；若支持吻合但实际安全失效，须处理安全模型可信度。即使实际安全
且终点改善，也必须同时满足完整原caps和独立物理资格后才能推进真实控制。
