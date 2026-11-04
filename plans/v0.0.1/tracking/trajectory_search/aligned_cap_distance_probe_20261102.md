# reserve下限对齐的一次完整联合方向探针

2026-11-02，seed42，按8358769设计，新显式cap_distance_gn_with_reserve_entry_floor
仅在pose阶段收紧模型safety floor至至少原reserve_entry_fraction，原实际门不变。
GPU禁用，CPU only，不修改旧默认/失败记录/参考或原18/12/24稳定验收。

使用同一anchor85及f4bc1804新Js/Jq/Je、原根global/pose ceiling、完整cap-distance
B/g及原float64算术。独立重建新floor及B/g/A/b逐值比较，原terminal单位保持。
QP前登记矩阵、源码/输入SHA；1QP、最多100迭代/1NNLS，原1e-6证书、scaling/
dual recovery审核。不合格/任一command预检失败则终止，不换方向或追加预算。

固定alpha1/.5/.25/.125/.0625，全部先冻结，anchor+5trial+repeat生产7响应/
2100内步/2185查询，独立7/2100/2185，共14/4200/4370、2model，0新模型拟合。
整体1QP/实际NNLS单列，0物理/GPU/WAM/fresh/最终留出/实际参考消费，候选不下发。
原1e-10/torque1e-8和布尔审核、完整响应、历史anchor/repeat exact，QA/source/map/
driver绑定，源码/输入前后SHA。CPU空CUDA/LD、数值库单线程，600秒TERM/10秒KILL，
实际wait/reap，无重试。不覆盖旧结果或追加任何旧60stage搜索预算。

主要记录24caps及全部实际安全，另记loss下降/预测比、reserve及原后续模式。
模型满足新floor不保证实际reserve≥.9，仍如实拒绝，不改实际门。一次新方向
证据不能替代连续有限迭代或物理/任务成功，后续预算另登记。
