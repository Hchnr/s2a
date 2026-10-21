# 新发现15600完整checkpoint的state对照 — 2026-10-21（Asia/Shanghai）

状态：固定单候选48次state-only预算，未授予物理评测或新初态性能。此前15400报告对应当时冻结身份，保留原字节；只读重新检查发现同23-source FastUMI run的latest_checkpoint现指iter15600且有.save_complete。latest不是已知best，不能因iteration更大预设学习质量更好。

仅对该新完整15600做原始WAM state筛查：与8400/8600/8800/15400 guidance1.0一致的原24 diagnostic样本、6旧episode/22不同anchor、H40/20Hz、task-only、采样seed42/43、35步/shift5，共48模型请求、0物理步。输入数组/PNG及有效mask保留原SHA，重复phase照原请求执行并另注明44独特anchor-seed；不打开预留新初态或重新挑样本。g1以已有固定合同为依据，不额外扩guidance/采样步矩阵。

模型仅物理GPU6/port9166；CPU编排CUDA_VISIBLE_DEVICES与LD_LIBRARY_PATH为空。GPU4/5/7现有005/006仍按原源码/配置完成。启动前完整读取15600 model全部文件、run config及.save_complete的SHA、大小、mtime，确认身份/非EMA及同15400 action128合同，绑定小文件来源审计；列明查重范围，无完全相同既有结果才执行。不修改外部Cosmos或发布数据，不重新hash旧32GB权重。

复用现有通用screen/state执行入口及mask wrapper，不改变live依赖。样本manifest复制原诊断cases，仅更新对应config/output；生成前后检查权重stat和self-owned进程/卡/端口，记录PID/starttick。全部48完整后独立从原始响应重建位置/SO3/开度及H8/H16/H40，保持valid future、不把padding/seed当独立episode，比较hold/恒速及旧stage。来源、代码、每请求/输入/完整权重指纹与正常release保留。

本预算不做校正器训练、任务规划、物理或新初态验收。若新stage无state证据优势，不因为其是latest自动扩大物理；后续短requery候选仍须结合005/006完整结果和007资格另行冻结。
