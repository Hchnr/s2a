# 8400更强文本引导的state诊断 — 2026-10-21（Asia/Shanghai）

状态：新增单候选state-only预算，在任何请求前冻结；不改变010/011和新初态验收。8400原raw开发控制尚无稳定多任务能力，文本条件是否足够约束生成仍有待验证；guidance1和1.5已有其它stage对照，不据此认定guidance3一定有效。本试验只检验单个适度更强CFG值3.0，禁止中途扩展参数网格。

固定8400非EMA、active十槽mask、35采样步/shift5/H40/20Hz、task-only、与原8400_g10完全同24 diagnostic样本/6旧episode/22独特anchor及seed42/43，48个新H40请求、0物理步。沿用同mask/归一化/相机/domain/首帧输入，原始full128和全部响应完整保存。仅guidance变为3.0。模型仅物理GPU7/port9577单卡/单rank/单worker；CPU编排CUDA/LD为空，GPU/port锁内核空闲、不停止其他进程。

复用冻结通用run_wam_stability_screen、serve_wam_acceptance、state_matrix，不改任何运行中的源。准备样本只复制原24，不接触fresh性能；按既有完整8400权重path/size/mtime复核，不重复读取旧32GB。生成前后记录实际源码、预算/launch提交、输入SHA、权重身份、mask、非EMA、PID/startticks和正常release。已有输出拒绝覆盖，一次基础设施失败保存停止、不自动重试。

与缓存g1对比全部H8/H16/H40位置/SO3/opening、hold/恒速及逐episode/phase，不把seed/重复phase当独立episode。此入口不记录实际initial noise witness，因此相同seed仅是固定请求条件；不将该比较冒称已验证actual-noise bitexact的单变量数值实验。011的actual-noise采样步数研究与此独立。

后续物理候选门槛在运行前固定为：H8/H16六项均不差于原8400_g10（数值比较仅容1e-10绝对误差），且至少一个位置RMSE改善10%；不满足就不追加guidance3物理。即使满足也须另登记完整多任务开发预算，不能中途替换010或自动打开新初态。指导增强不是修复已证实错误，state优势也不等于task成功。本预算不新增训练/参考校正或任务规划。
