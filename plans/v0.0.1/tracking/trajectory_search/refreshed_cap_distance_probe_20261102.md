# 完整刷新局部模型的一次联合目标验证

2026-11-02，seed42，CPU only。使用已实际重建并独立审核的anchor85联动模型，
不新拟合、不修改参考/接口/目标门槛，GPU禁用。18/12/24及最终稳定验收保持。

固定读取bee4d939结果及f4bc1804完整新stencil的Js/Jq/Je和base，原global bounds、
pose ceiling、reserve目标/模式切换、安全buffer、.01m/.05rad caps不变。
完整cap-distance GN目标及原float64算术保持。独立重建B/g/A/b逐值比较；新Js/Jq
会改变线性系数，不将这些系数变化误称安全门放宽。

1QP、最多100迭代、最多1NNLS，QP前保存完整矩阵和输入SHA，原1e-6证书/
objective scaling/dual recovery审核。未合格或任一原command门失败即终止，
不换方向、不裁剪、不补选。固定alpha1/.5/.25/.125/.0625，全部先冻结再执行，
不按前几个结果修改列表。沿用已经测试的fixed_trials及共享完整响应runtime。

anchor+5trial+repeat生产7响应/2100内步/2185查询；独立7/2100/2185，
共14响应/4200内步/4370查询、2model。0新拟合，整体1QP/实际NNLS次数另计，
runtime内部0QP不能冒充整体0QP。0物理/GPU/WAM/fresh/最终留出/实际参考消费。

空CUDA/LD、每数值库线程1，QA/source/map/driver绑定，原1e-10/torque1e-8布尔
审核及完整响应、anchor/repeat exact；来源/源码SHA前后不变。600秒TERM/10秒KILL，
实际wait/reap，绝不下发任何候选，不覆盖旧失败或追加原搜索预算。

主要报告完整安全与24caps，另记loss预测/实际/比值、reserve及下轮原模式，
不能以训练残差小或单步安全下降替代完整跟踪。所有尺度均保留，后验分类不是新
搜索执行。只有完整新响应证据足够后，才另行规划恢复/迭代及物理验证。
