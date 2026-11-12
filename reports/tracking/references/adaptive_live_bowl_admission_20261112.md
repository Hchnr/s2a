# 连续重规划：已登记历史与并发启动准入

2026-11-12。试验准入固定配置和计划字节，重新核对 step107 状态、图像、
同图模型响应、前次独立重放及累计开销，登记 1004 个模型源码文件和 163 个
权重文件的待预检清单。checker 加载准入时不导入候选策略。

修正并发启动时序：模型 worker 验证预先存在的仿真父目录，具体 runtime 子目录
由仿真进程独占创建；因此模型先启动时不会因子目录尚不存在而误报。32 并发相关
回归 **45 项通过、0 skip、0 collection errors**；检查记录
`outputs/qa/adaptive_live_bowl_admission_race_20261112/result.json` SHA-256
`fdd540e568edf86c69afcb356b0b180944cb2731e16f9a9ac030fcb153bbdeb8`。
本阶段无新模型初始化、GPU、物理仿真或留出试验。
