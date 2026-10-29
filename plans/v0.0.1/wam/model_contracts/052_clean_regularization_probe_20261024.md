# 052：固定单值CPU正则化对照

目标沿用036；当前W2性能失败，fresh保持关闭。此次仅检查“全向量L2归一化下ridge过强”这一待验证假设，不能把teacher拟合改进等同任务成功。

依据：完整840000行特征/target流SHA复现原W2，现有头训练归一化xyz MSE改善2.9597%、opening改善1.9940%、rot6d改善42.7172%；实际特征有效自由度5.2947。原选择并非数值实现错误，原资格和失败记录保留。详见reports/wam_clean_capacity_diagnosis_20261024.md。

固定唯一新值：global-vector-L2归一化下ridge=0.1/1290。这在精确算术下等价于采用全局逐元素RMS归一化、保持原0.1 feature惩罚，并将bias惩罚相应调整；**不是**保持相同bias惩罚的全目标等价换写。这里明确同时把旧bias惩罚从0.1降至0.1/1290，故对照含feature与bias两项变化，不能唯一归因feature容量。保留全向量归一化和完整所有1290列，避免静默改变数值接口。单值选择来自维度尺度约定，不按task、seed或validation结果搜索。

预算：同一已完成D0/D1缓存、150训练锚点、600请求/840000行，CPU生产fit最多1次，双遍归一化和单次ridge solve；0新模型、重采、物理和fresh。独立数值重构另记录，不能与producer同函数比一次参数便称独立QR通过。输出outputs/clean_regularization_probe_20261024，不写原fit或任何051冻结来源。新参数不得用旧W2身份或旧注册envelope部署。

执行前绑定原fit/params/训练prepared/normalization、生产脚本与原streaming/readout源码SHA；完整每遍feature、target、weight流须与原fit完全一致。拒绝不完整原fit、数据漂移、重复输出、非空CUDA/LD或非显式8线程CPU环境。不覆盖失败或自动重试。拟合完成记录actual fit attempt1及result；未完成独立数值审查不得宣称训练资格完成。

后续顺序：独立重构与teacher误差分层；固定独立validation短期位置/SO(3)/opening保护和双seed门槛后，才另建新模型采样资格。通过后再统一原三任务×init0×seed42/43固定开发，不拼接策略、换初态或追加重试；开发≥5/6且各任务≥1/2才具备进一步冻结条件，最终原24条≥20/24及各任务≥4/8保持。旧STATE初态同步未认证边界保留，新资格必须明确解决，不能复用间接证据当成通过。
