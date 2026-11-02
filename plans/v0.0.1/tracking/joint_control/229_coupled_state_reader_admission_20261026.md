# 229 当前M/J读取与旋转goal的原生环境准入

228核心10项CPU测试已覆盖世界SO3弹簧导数、独立ODE响应、无耦合退化085、耦合LS/KKT、实际命令/完成后创新与重置，以及零旋转锁存。现在按226零积分保护重建原case15/06/14的step550/220/60端点。

恢复原已记录当前controller goal缓存（仅恢复，不调用set_goal/不执行action），核对read_current_dynamics从实际当前robot q/qvel/SITE取得M/J，与当前controller更新值和独立原生API一致。读取前后sim_state、ctrl、sensor不得改变；陈旧joint packet和不支持的uncoupling实例必须拒绝。候选旋转只使用当前命令、当前SITE与当前previous_goal，核对其native目标与原日志持有goal标签一致；这个归档标签仅用于验证，不作输入。

在该当前矩阵/rotation goal上构建耦合响应和单步平移逆，独立SO3差分+ODE重建响应，并对JSON往返结果检查相同2e-6 KKT与边界，不执行候选命令。实际task_id由manifest构造expected集合，要求三个不同任务，不误用展示编号。零mj_step/mj_step2；允许状态与ctrl逐次完全不变的初始化mj_step1，并保存计数和版本/来源。通过只为当前输入/数学准入，非控制成功。
